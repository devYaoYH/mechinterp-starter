"""Activations from *sampled generation*, not a single teacher-forced pass.

`activations.py` assumes one deterministic forward pass over a fixed string.
That is the wrong distribution for anything that has to work during generation:
a live gate, or a direction meant to steer a model mid-answer. Training a
direction on the last token of a complete, well-formed statement and then
applying it to the last token of an in-progress phrase (tokens like "The",
"capital", "of", which carry no claim yet) is a train/deploy mismatch, and the
resulting behaviour is uncalibrated noise rather than a weak signal.

This module captures what the model actually does when it generates: per-step
activations at every layer, aligned to the token that was emitted, over sampled
rollouts, with an auto-verifier so each rollout carries a correctness label.
That is the distribution a rollout-trained direction or gate needs.

    rs = sample_rollouts(model, tok, prompts, verifier=contains("Paris"),
                         n_samples=4, temperature=0.8)
    X, y, pos = stack_steps(rs, layer=15, relative_to="end")
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch


@dataclass
class Rollout:
    prompt: str
    text: str                      # generated continuation only
    token_ids: list
    token_texts: list
    acts: np.ndarray               # [n_steps, n_layers+1, hidden]
    correct: Optional[bool] = None
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.token_ids)


def contains(answer, case_sensitive=False):
    """Simple verifier factory: did the generation contain this string?"""
    def _v(text, _prompt=None):
        return (answer in text) if case_sensitive else (answer.lower() in text.lower())
    return _v


@torch.no_grad()
def sample_rollouts(model, tok, prompts, max_new_tokens=24, n_samples=1,
                    temperature=1.0, do_sample=True, seed=0,
                    verifier=None, answers=None):
    """Generate with sampling and keep the per-step residual stream.

    prompts: list of strings, already chat-templated if that is what you want
    (apply tok.apply_chat_template yourself -- keeping it out of here means one
    less thing to unpick when your prompt format differs).
    answers: optional per-prompt expected answer, passed to
    `verifier(text, answer)` when the verifier takes two arguments.

    Returns a list of Rollout. Uses generate(output_hidden_states=True), whose
    hidden_states is a tuple over STEPS, each a tuple over layers; step 0 is the
    prefill (seq = prompt length) and later steps have seq = 1. We take the last
    position of each step, which is the state that produced that step's token.
    """
    torch.manual_seed(seed)
    out_rollouts = []
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").to(model.device)
        n_in = ids["input_ids"].shape[1]

        for s in range(n_samples):
            gen = model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tok.eos_token_id,
                return_dict_in_generate=True, output_hidden_states=True)
            new_ids = gen.sequences[0, n_in:].tolist()
            steps = []
            for step_hs in gen.hidden_states:          # one entry per generated token
                steps.append(torch.stack([h[0, -1] for h in step_hs]).float().cpu().numpy())
            acts = np.stack(steps[:len(new_ids)]) if new_ids else np.zeros((0, 1, 1))
            text = tok.decode(new_ids, skip_special_tokens=True)

            correct = None
            if verifier is not None:
                ans = answers[i] if answers is not None else None
                try:
                    correct = bool(verifier(text, ans))
                except TypeError:
                    correct = bool(verifier(text))
            out_rollouts.append(Rollout(
                prompt=prompt, text=text, token_ids=new_ids,
                token_texts=[tok.decode([t]) for t in new_ids],
                acts=acts, correct=correct,
                meta={"sample": s, "prompt_index": i, "temperature": temperature}))
    return out_rollouts


def stack_steps(rollouts, layer, relative_to="start", only_labeled=True,
                max_offset=None):
    """Flatten rollouts into a probe-ready matrix, one row per generated token.

    relative_to: "start" -> position index counts from the first generated token;
                 "end"   -> negative offsets from the last generated token, which
                            is the alignment you want when generations differ in
                            length but all end the same way.

    -> X [n_rows, hidden], y [n_rows] (rollout.correct), pos [n_rows],
       group [n_rows] (rollout index -- pass to probing.grouped_split so tokens
       from one rollout cannot straddle the train/test split).
    """
    X, y, pos, grp = [], [], [], []
    for gi, r in enumerate(rollouts):
        if only_labeled and r.correct is None:
            continue
        n = len(r)
        if n == 0:
            continue
        for t in range(n):
            offset = t if relative_to == "start" else t - n
            if max_offset is not None and abs(offset) > max_offset:
                continue
            X.append(r.acts[t, layer + 1])          # layer L == slot L+1
            y.append(int(bool(r.correct)))
            pos.append(offset)
            grp.append(gi)
    if not X:
        raise ValueError("no rows -- were the rollouts labeled by a verifier?")
    return np.stack(X), np.array(y), np.array(pos), np.array(grp)


def summarize(rollouts):
    n = len(rollouts)
    labeled = [r for r in rollouts if r.correct is not None]
    acc = np.mean([r.correct for r in labeled]) if labeled else float("nan")
    lens = [len(r) for r in rollouts]
    lines = [f"{n} rollouts, {len(labeled)} labeled, accuracy {acc:.3f}",
             f"  tokens/rollout: min {min(lens)}, median {int(np.median(lens))}, max {max(lens)}"]
    if labeled and 0.02 < acc < 0.98:
        lines.append("  usable error rate -- suitable for training a rollout-based probe")
    elif labeled:
        lines.append("  WARNING: accuracy is near 0 or 1; there are too few of one class "
                     "to train on. Make the task harder or easier before probing.")
    return "\n".join(lines)
