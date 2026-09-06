"""Linear probing, with the controls attached to the measurement.

The API deliberately returns the probe score and its controls together, because
an uncontrolled probe number is not interpretable and is very easy to report by
accident. Three things go wrong in practice, in rough order of how often:

1. n_train < n_features. Residual streams are 1.5k-8k dimensional; a few hundred
   examples are ALWAYS linearly separable, so a high score means nothing.
   `probe()` warns when you are in that regime.
2. The label is a deterministic function of something trivially present in the
   input. A probe for "which capital is coming" scores well by decoding "which
   country was mentioned" -- no retrieval required. Pass `leak_X/leak_y` with
   examples where the shortcut is present but the label should NOT follow, and a
   high `leak_score` tells you the probe took it.
3. Grouped structure leaked across the split -- e.g. multiple token positions
   from the same sentence in both train and test. Split by group, not by row.

`shuffled_score` is the null: fit the same pipeline on permuted labels. If it is
not near chance, the evaluation itself is broken.
"""
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _pipeline(n_train, n_features, C=0.1, max_pca=120):
    """StandardScaler is not optional: residual streams have a few huge-norm
    outlier dimensions, and unscaled features make C mean something different
    per dimension. PCA keeps the fit fast and cannot lose signal a linear probe
    could have used, since the data already spans <= n_train dimensions."""
    steps = [StandardScaler()]
    n_comp = min(max_pca, max(2, n_train - 1), n_features)
    steps.append(PCA(n_components=n_comp, random_state=0))
    steps.append(LogisticRegression(max_iter=3000, C=C))
    return make_pipeline(*steps)


def probe(X_train, y_train, X_test, y_test, leak_X=None, leak_y=None, C=0.1,
          metric="accuracy"):
    """Fit a linear probe and its controls in one call.

    leak_X / leak_y: inputs where a shortcut feature is still present but the
    label should not follow from it. High leak_score == the probe is reading
    the shortcut. See the module docstring.

    -> dict(score, shuffled_score, leak_score, chance, n_train, n_features,
            underdetermined, per_item)
    `per_item` is the 0/1 correctness vector on the test set (accuracy metric
    only) -- feed it to plotting.bootstrap_ci for an interval on the score.
    """
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    n_train, n_features = X_train.shape
    classes = np.unique(y_train)
    chance = 1.0 / len(classes) if metric == "accuracy" else 0.5

    underdetermined = n_features >= n_train
    if underdetermined:
        warnings.warn(
            f"n_features ({n_features}) >= n_train ({n_train}): the training set "
            f"is always linearly separable in this regime, so treat the held-out "
            f"score as the only meaningful number and expect it to be noisy.",
            stacklevel=2)

    def _fit_score(ytr, Xte, yte):
        clf = _pipeline(n_train, n_features, C=C).fit(X_train, ytr)
        if metric == "auroc":
            return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), None
        return clf.score(Xte, yte), (clf.predict(Xte) == yte).astype(float)

    score, per_item = _fit_score(y_train, X_test, y_test)
    shuffled, _ = _fit_score(np.random.RandomState(0).permutation(y_train), X_test, y_test)

    leak = None
    if leak_X is not None:
        clf = _pipeline(n_train, n_features, C=C).fit(X_train, y_train)
        leak = float((clf.predict(np.asarray(leak_X)) == np.asarray(leak_y)).mean())

    return dict(score=float(score), shuffled_score=float(shuffled), leak_score=leak,
                chance=chance, n_train=int(n_train), n_features=int(n_features),
                underdetermined=bool(underdetermined), per_item=per_item)


def grouped_split(groups, test_frac=0.3, seed=0):
    """Split by group id, never by row. Use the example/sentence/template id as
    the group so the same underlying item cannot appear on both sides."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    test = set(uniq[:max(1, int(len(uniq) * test_frac))].tolist())
    is_test = np.array([g in test for g in groups])
    return ~is_test, is_test


def format_result(r, label=""):
    lead = f"{label:<16}" if label else ""
    leak = "  leak=  n/a" if r["leak_score"] is None else f"  leak={r['leak_score']:6.3f}"
    flag = "  [UNDERDETERMINED]" if r["underdetermined"] else ""
    return (f"{lead}score={r['score']:6.3f}  shuffled={r['shuffled_score']:6.3f}"
            f"  chance={r['chance']:5.3f}{leak}{flag}")
