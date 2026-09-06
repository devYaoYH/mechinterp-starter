"""Activations from sampled generation, not one teacher-forced pass.

activations.py assumes a fixed string. That is the wrong distribution for
anything that must work *during* generation: a direction fit on the last token
of a complete statement, then applied to an in-progress phrase ("The",
"capital", "of" -- tokens with no claim in them yet), is a train/deploy mismatch
and behaves as noise, not weak signal.

    rs = sample_rollouts(model, tok, prompts, verifier=contains("Paris"),
                         n_samples=4, temperature=0.8)
    X, y, pos, grp = stack_steps(rs, layer=15, relative_to="end")
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Rollout:
    prompt: str
    text: str                    # generated continuation only
    token_ids: list
    token_texts: list
    acts: np.ndarray             # [n_steps, n_layers + 1, hidden]
    correct: bool = None
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.token_ids)


def contains(answer, case_sensitive=False):
    """Verifier factory: did the generation contain this string?"""
    def check(text, _answer=None):
        return answer in text if case_sensitive else answer.lower() in text.lower()
    return check


@torch.no_grad()
def sample_rollouts(model, tok, prompts, max_new_tokens=24, n_samples=1,
                    temperature=1.0, do_sample=True, seed=0, verifier=None,
                    answers=None):
    """Generate with sampling, keeping the per-step residual stream.

    prompts: plain strings -- apply a chat template yourself if you want one.
    verifier: called as verifier(text, answer) or verifier(text) to label each
    rollout; answers supplies the per-prompt expected value.

    generate(output_hidden_states=True) yields hidden_states as a tuple over
    STEPS, each a tuple over layers; step 0 is prefill (seq = prompt length),
    later steps have seq = 1. We take the last position of each step -- the
    state that produced that step's token.
    """
    torch.manual_seed(seed)
    out = []
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").to(model.device)
        n_in = ids["input_ids"].shape[1]
        for s in range(n_samples):
            gen = model.generate(**ids, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample,
                                 temperature=temperature if do_sample else None,
                                 pad_token_id=tok.eos_token_id,
                                 return_dict_in_generate=True, output_hidden_states=True)
            new_ids = gen.sequences[0, n_in:].tolist()
            steps = [torch.stack([h[0, -1] for h in step]).float().cpu().numpy()
                     for step in gen.hidden_states]
            text = tok.decode(new_ids, skip_special_tokens=True)
            correct = None
            if verifier is not None:
                ans = answers[i] if answers is not None else None
                try:
                    correct = bool(verifier(text, ans))
                except TypeError:
                    correct = bool(verifier(text))
            out.append(Rollout(prompt, text, new_ids, [tok.decode([t]) for t in new_ids],
                               np.stack(steps[:len(new_ids)]) if new_ids else np.zeros((0, 1, 1)),
                               correct, {"sample": s, "prompt_index": i}))
    return out


def stack_steps(rollouts, layer, relative_to="start", only_labeled=True, max_offset=None):
    """Flatten to probe rows, one per generated token.

    relative_to: "start" counts from the first generated token; "end" gives
    negative offsets from the last -- the alignment you want when generations
    differ in length but end the same way.

    -> X [n, hidden], y [n], pos [n], group [n]. Pass `group` to
    probing.grouped_split so one rollout's tokens cannot straddle the split.
    """
    X, y, pos, grp = [], [], [], []
    for gi, r in enumerate(rollouts):
        if (only_labeled and r.correct is None) or not len(r):
            continue
        for t in range(len(r)):
            offset = t if relative_to == "start" else t - len(r)
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
    labeled = [r for r in rollouts if r.correct is not None]
    acc = float(np.mean([r.correct for r in labeled])) if labeled else float("nan")
    lens = [len(r) for r in rollouts]
    out = [f"{len(rollouts)} rollouts, {len(labeled)} labeled, accuracy {acc:.3f}",
           f"  tokens/rollout: min {min(lens)}, median {int(np.median(lens))}, max {max(lens)}"]
    if labeled and not 0.02 < acc < 0.98:
        out.append("  WARNING: accuracy near 0 or 1 -- too few of one class to train on. "
                   "Make the task harder or easier before probing.")
    return "\n".join(out)
