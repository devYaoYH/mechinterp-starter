"""Linear probing, with the controls attached to the measurement.

`probe()` returns the score together with its controls, because an uncontrolled
probe number is not interpretable. What goes wrong, in rough order of frequency:

1. n_features >= n_train. Residual streams are 1.5k-8k dimensional; a few hundred
   examples are ALWAYS separable, so a high training score is guaranteed.
   -> `underdetermined` flag.
2. The label is a deterministic function of something trivially in the input.
   "Which capital is coming" is the same label partition as "which country was
   mentioned", so a probe scores ~100% at layer 0 having retrieved nothing.
   -> `leak_score`: inputs where the shortcut is present but the label should not
   follow. High == the probe took it.
3. The probe only works on the distribution it was fit on.
   -> `transfer_score`: held-out data from a different domain/category/group.
4. Grouped structure straddling the split (token positions from one sentence on
   both sides). -> `grouped_split`, by item id, never by row.

5. AUROC hides the corner you deploy in: a detector lives at a low false-alarm
   rate, and a probe at AUROC 0.92 can catch almost nothing at 1% FPR.
   -> `recall_at_fpr`, against a NEUTRAL CONTROL set. That set is a third thing:
   `leak` asks "did it take a shortcut", `transfer` asks "does it work
   elsewhere", control-FPR asks "how often does it fire on benign input".

`shuffled_score` is the null: same pipeline, permuted labels. If it is not near
chance, the evaluation itself is broken.
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
    dimensions, and unscaled features make C mean something different per
    dimension. PCA keeps fits fast and loses nothing a linear probe could use,
    since the data already spans <= n_train dimensions."""
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(max_pca, max(2, n_train - 1), n_features), random_state=0),
        LogisticRegression(max_iter=3000, C=C))


def recall_at_fpr(scores_positive, scores_control, fpr=0.01, n_boot=2000, seed=0):
    """Recall at a threshold allowing `fpr` false positives on the controls.

    The operating point AUROC averages away -- a probe at AUROC 0.92 can catch
    almost nothing at 1% FPR. Report both.

    The threshold is a QUANTILE OF THE CONTROLS, so it is only as good as that
    set is large: at 1% FPR, fewer than 100 controls cannot place it at all and
    the number is an artifact of the largest two or three. Hence the bootstrap
    CI over both sets, and the warning.

    -> dict(recall, threshold, ci, n_control, fpr, warning)
    """
    pos = np.asarray(scores_positive, float)
    ctl = np.asarray(scores_control, float)
    thr = float(np.quantile(ctl, 1 - fpr))

    # Resample BOTH sets: the threshold is an estimate too, and on a small
    # control set it is the larger source of uncertainty.
    rng = np.random.RandomState(seed)
    boot = [float((pos[rng.randint(0, pos.size, pos.size)] >
                   np.quantile(ctl[rng.randint(0, ctl.size, ctl.size)], 1 - fpr)).mean())
            for _ in range(n_boot)]

    need = int(np.ceil(1 / fpr))
    return dict(recall=float((pos > thr).mean()), threshold=thr,
                ci=(float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
                n_control=int(ctl.size), fpr=float(fpr),
                warning=None if ctl.size >= need else
                f"{ctl.size} controls cannot place a {fpr:.0%} FPR threshold "
                f"(need >= {need}); it is just the control maximum")


def probe(X_train, y_train, X_test, y_test, leak_X=None, leak_y=None,
          transfer_X=None, transfer_y=None, control_X=None, fpr=0.01,
          C=0.1, metric="accuracy"):
    """-> dict(score, shuffled_score, leak_score, transfer_score, recall_at_fpr,
              chance, n_train, n_features, underdetermined, per_item)

    `per_item` is the 0/1 correctness vector on the test set (accuracy metric
    only); feed it to plotting.bootstrap_ci for an interval on the score.

    `control_X` is a NEUTRAL set -- inputs that should not trip the probe, all
    negatives by definition, so it carries no labels. Binary probes only.
    """
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    n_train, n_features = X_train.shape
    chance = 1.0 / len(np.unique(y_train)) if metric == "accuracy" else 0.5

    underdetermined = n_features >= n_train
    if underdetermined:
        warnings.warn(f"n_features ({n_features}) >= n_train ({n_train}): the training "
                      f"set is always separable here, so the held-out score is the only "
                      f"meaningful number, and it will be noisy.", stacklevel=2)

    def fit_score(ytr):
        clf = _pipeline(n_train, n_features, C).fit(X_train, ytr)
        if metric == "auroc":
            return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]), None, clf
        return clf.score(X_test, y_test), (clf.predict(X_test) == y_test).astype(float), clf

    score, per_item, clf = fit_score(y_train)
    shuffled, _, _ = fit_score(np.random.RandomState(0).permutation(y_train))

    def apply(Xo, yo):
        """Score a control set with the SAME metric as the headline number.
        Reporting an AUROC score next to an accuracy leak_score invites reading
        them as comparable, which they are not."""
        if Xo is None:
            return None
        Xo, yo = np.asarray(Xo), np.asarray(yo)
        if metric == "auroc":
            if len(np.unique(yo)) < 2:
                return None        # AUROC is undefined on a single-class set
            return float(roc_auc_score(yo, clf.predict_proba(Xo)[:, 1]))
        return float((clf.predict(Xo) == yo).mean())

    recall = None
    if control_X is not None:
        if len(np.unique(y_train)) != 2:
            raise ValueError("recall_at_fpr is defined for binary probes only")
        # clf.classes_[1] is the class decision_function scores POSITIVE;
        # picking it by np.max(y_test) breaks on a single-class test set.
        pos = clf.decision_function(X_test)[y_test == clf.classes_[1]]
        recall = recall_at_fpr(pos, clf.decision_function(np.asarray(control_X)), fpr)

    return dict(score=float(score), shuffled_score=float(shuffled),
                leak_score=apply(leak_X, leak_y),
                transfer_score=apply(transfer_X, transfer_y),
                recall_at_fpr=recall,
                chance=chance, n_train=int(n_train), n_features=int(n_features),
                underdetermined=bool(underdetermined), per_item=per_item)


def grouped_split(groups, test_frac=0.3, seed=0):
    """Split by group id, never by row -> (train_mask, test_mask)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    np.random.RandomState(seed).shuffle(uniq)
    test = set(uniq[:max(1, int(len(uniq) * test_frac))].tolist())
    is_test = np.isin(groups, list(test))
    return ~is_test, is_test


def format_result(r, label=""):
    parts = [f"{label:<16}" if label else "",
             f"score={r['score']:6.3f}  shuffled={r['shuffled_score']:6.3f}",
             f"  chance={r['chance']:5.3f}"]
    for key, name in (("leak_score", "leak"), ("transfer_score", "transfer")):
        if r.get(key) is not None:
            parts.append(f"  {name}={r[key]:6.3f}")
    if r["underdetermined"]:
        parts.append("  [UNDERDETERMINED]")
    rf = r.get("recall_at_fpr")
    if rf:
        parts.append(f"\n{'':<16}recall@{rf['fpr']:.0%}FPR={rf['recall']:6.3f} "
                     f"[{rf['ci'][0]:.3f}, {rf['ci'][1]:.3f}]  "
                     f"(n_control={rf['n_control']})")
        if rf["warning"]:
            parts.append(f"\n{'':<16}WARNING: {rf['warning']}")
    return "".join(parts)
