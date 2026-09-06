"""Logit lens: project an intermediate residual stream through the model's own
final norm + unembedding, to ask "is token X promoted at this site yet?"

Untrained, so it is a LOWER BOUND on decodability -- "not visible to the lens"
is not "not represented". Its compensating virtue is having no trainable
parameters to launder a shortcut through, which is the failure mode a trained
probe on a small dataset is prone to. Use both; they bracket from opposite sides.
"""
import torch


def logit_lens(model, hidden, already_normed=False, lm_head=None, layer_norm=None):
    """already_normed: skip the final norm for a state that has had it applied.
    Getting this wrong does not raise -- it returns confident nonsense
    (' the', ' a', ' in' instead of the answer). Prefer `lens_at`."""
    with torch.no_grad():
        h = hidden.detach()
        if not already_normed:
            h = (layer_norm or model.model.norm)(h)
        return (lm_head or model.lm_head)(h).float()


def lens_at(model, hs, slot, position=-1):
    """Logit lens on `hs` from activations.all_layers, handling the final-norm
    bookkeeping. `slot` indexes hs directly, so slot = L + 1 for layer L."""
    return logit_lens(model, hs[slot][position], already_normed=(slot == hs.shape[0] - 1))


def token_rank(logits, token_id):
    """1 = top of vocabulary. Rank is the honest scalar for "is it here yet" --
    a raw logit is unreadable without the rest of the distribution."""
    return int((logits > logits[token_id]).sum().item()) + 1


def top_k(tok, logits, k=5):
    v, i = torch.topk(logits, k)
    return [(tok.decode([int(t)]), float(s)) for s, t in zip(v, i)]
