"""Logit lens: project an intermediate residual stream through the model's own
final norm + unembedding, to ask "is token X promoted at this site yet?"

Untrained, so it is a LOWER BOUND on decodability -- "not visible to the logit
lens" is not "not represented". Its compensating virtue is that it has no
trainable parameters to launder a shortcut through, which is exactly the
failure mode a trained probe on a small dataset is prone to (see probing.py).
Use both; they bracket the answer from opposite sides.
"""
import torch


def logit_lens(model, hidden, already_normed=False, layer_norm=None, lm_head=None):
    """hidden: [hidden] or [..., hidden] residual stream. -> logits.

    already_normed: skip the final norm because this state has had it applied
    already. See `lens_at` -- getting this wrong does not raise, it just
    returns confident nonsense (' the', ' a', ' in' instead of the answer).
    """
    head = lm_head if lm_head is not None else model.lm_head
    with torch.no_grad():
        h = hidden.detach()
        if not already_normed:
            h = (layer_norm if layer_norm is not None else model.model.norm)(h)
        return head(h).float()


def lens_at(model, hs, slot, position=-1):
    """Logit lens on `hs` from activations.all_layers, doing the final-norm
    bookkeeping for you. `slot` indexes hs directly, so slot = L + 1 for
    decoder layer L. Prefer this over calling logit_lens by hand."""
    return logit_lens(model, hs[slot][position],
                      already_normed=(slot == hs.shape[0] - 1))


def token_rank(logits, token_id):
    """1 = top of the vocabulary. Rank is the honest scalar for 'is it here
    yet' -- a raw logit is unreadable without the rest of the distribution."""
    return int((logits > logits[token_id]).sum().item()) + 1


def top_k(tok, logits, k=5):
    v, i = torch.topk(logits, k)
    return [(tok.decode([int(t)]), float(s)) for s, t in zip(v, i)]
