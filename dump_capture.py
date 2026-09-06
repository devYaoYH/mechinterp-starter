"""Gate a capture before spending anything downstream.

Probe training, patching sweeps and steering runs are expensive and, worse,
they return plausible numbers on a broken dataset. The failures that matter
here do not raise: tokens misaligned, a subject index on the wrong token, rows
that are byte-identical across labels (so the task is degenerate by
construction), one class nearly absent, too few groups to split by.

This dumps a capture -- static prompts OR sampled rollouts -- to self-contained
JSON, runs the gate checks, and prints a verdict. Drag the JSON into the viewer
for the interactive version.

    python dump_capture.py --out capture.json            # static demo
    python dump_capture.py --out roll.json --rollouts    # rollout demo

Adapting it: build `items` yourself and call `build_capture` / `gate`.
Both capture kinds share one row model -- an item has 1 row (static) or one row
per generated token (rollout) -- so every check below applies to both.
"""
import hashlib
import json
import os

import numpy as np

from mechinterp import activations as A
from mechinterp import loading, readout, rollout


def _hash(vec):
    """Digest of a rounded vector: spots rows identical at this site despite
    carrying different labels."""
    return hashlib.sha1(np.round(np.asarray(vec, float), 3).tobytes()).hexdigest()[:12]


def _row(vec_by_slot, pos, token):
    return {"pos": pos, "token": token,
            "norms": [round(float(np.linalg.norm(v)), 4) for v in vec_by_slot],
            "hashes": [_hash(v) for v in vec_by_slot],
            "finite": bool(np.isfinite(vec_by_slot).all())}


def build_capture(model, tok, items, pool="last", lens_k=5):
    """Static capture. items: [{prompt, label?, group?, marks?}] where `marks`
    is {name: word}, each resolved to a token index so the viewer can show what
    it actually landed on."""
    NL = loading.n_layers(model)
    slots = sorted({0, NL // 4, NL // 2, 3 * NL // 4, NL})
    out, vecs = [], []
    for it in items:
        hs = A.all_layers(model, tok, it["prompt"])
        p = A.pool_tokens(hs, pool).float().cpu().numpy()
        toks = [tok.decode([t]) for t in tok(it["prompt"])["input_ids"]]
        marks = {}
        for name, word in (it.get("marks") or {}).items():
            try:
                i = A.token_index(tok, it["prompt"], word)
                marks[name] = {"index": i, "token": toks[i], "wanted": word,
                               "ok": toks[i].strip() == word}
            except ValueError as e:
                marks[name] = {"index": None, "wanted": word, "ok": False, "error": str(e)}
        out.append({"prompt": it["prompt"], "tokens": toks, "label": it.get("label"),
                    "group": it.get("group"), "marks": marks,
                    "rows": [_row(p, -1, toks[-1])],
                    "lens": {str(s): [t for t, _ in readout.top_k(
                        tok, readout.lens_at(model, hs, s), lens_k)] for s in slots}})
        vecs.append(p)
    return _finish("static", model, NL, np.stack(vecs), out, pool, slots)


def build_rollout_capture(rollouts, n_layers, group_by_prompt=True):
    """Rollout capture from mechinterp.rollout.sample_rollouts output. One row
    per generated token; `pos` is the offset from the END of the generation,
    which is the alignment that survives varying lengths."""
    out, vecs = [], []
    for gi, r in enumerate(rollouts):
        n = len(r)
        rows = [_row(r.acts[t], t - n, r.token_texts[t]) for t in range(n)]
        out.append({"prompt": r.prompt, "tokens": r.token_texts,
                    "generated": r.text, "label": None if r.correct is None else int(r.correct),
                    "group": r.meta.get("prompt_index") if group_by_prompt else gi,
                    "marks": {}, "rows": rows, "lens": {}})
        vecs.extend(r.acts[t] for t in range(n))
    return _finish("rollout", None, n_layers, np.stack(vecs) if vecs else np.zeros((0, 1, 1)),
                   out, "per-step", [])


def _finish(kind, model, NL, V, items, pool, slots):
    labels = [it["label"] for it in items for _ in it["rows"]]
    classes = sorted({x for x in labels if x is not None})
    sep = None
    if len(classes) == 2 and V.size:
        m = np.array(labels, dtype=object)
        a, b = V[m == classes[0]], V[m == classes[1]]
        # ||mu_a - mu_b|| / median ||h||, per slot. A cheap "is there anything
        # here at all" preview -- NOT a probe; it controls for nothing.
        sep = [round(float(np.linalg.norm(a[:, s].mean(0) - b[:, s].mean(0))
                           / (np.median(np.linalg.norm(V[:, s], axis=1)) + 1e-9)), 4)
               for s in range(V.shape[1])]
    cap = {"schema": "mechinterp-capture/2", "kind": kind,
           "model": getattr(getattr(model, "config", None), "_name_or_path", "unknown"),
           "n_layers": NL, "hidden_size": int(V.shape[-1]) if V.size else 0,
           "pool": pool, "lens_slots": slots, "items": items, "separation": sep}
    cap["checks"] = gate(cap)
    return cap


def gate(cap):
    """Run the gate checks -> [{name, status: pass|warn|fail, detail}]."""
    items = cap["items"]
    rows = [(it, r) for it in items for r in it["rows"]]
    labels = [it["label"] for it, _ in rows]
    known = [x for x in labels if x is not None]
    groups = {it.get("group") for it in items}
    checks = []

    def add(name, ok, warn, detail):
        checks.append({"name": name, "detail": detail,
                       "status": "pass" if ok else ("warn" if warn else "fail")})

    n_bad = sum(1 for _, r in rows if not r["finite"])
    add("finite activations", n_bad == 0, False,
        f"{len(rows) - n_bad}/{len(rows)} rows finite"
        + (f" -- {n_bad} contain NaN/Inf" if n_bad else ""))

    if not known:
        add("labels present", False, True, "no rows carry a label -- nothing to train on")
    else:
        counts = {c: known.count(c) for c in sorted(set(known))}
        minority = min(counts.values()) / len(known)
        add("label balance", minority >= 0.2, minority >= 0.05,
            f"{counts}, minority class {minority:.1%}"
            + ("" if minority >= 0.2 else " -- too few of one class to learn from"))

    # Rows identical at the final slot but labelled differently: the task is
    # degenerate at this site by construction, and no probe can beat chance.
    by_hash = {}
    for it, r in rows:
        by_hash.setdefault(r["hashes"][-1], set()).add(it["label"])
    clashes = sum(1 for v in by_hash.values() if len(v) > 1)
    add("no degenerate rows", clashes == 0, False,
        f"{clashes} activation values shared across different labels"
        + (" -- identical inputs cannot carry label information" if clashes else ""))

    n_marks = sum(len(it["marks"]) for it in items)
    bad_marks = [(it["prompt"], n, m) for it in items for n, m in it["marks"].items()
                 if not m.get("ok")]
    if n_marks:
        add("token marks resolve", not bad_marks, False,
            f"{n_marks - len(bad_marks)}/{n_marks} marks landed on the intended token"
            + (f" -- first failure: {bad_marks[0][1]!r} in {bad_marks[0][0]!r}" if bad_marks else ""))

    add("enough groups to split", len(groups) >= 5, len(groups) >= 3,
        f"{len(groups)} distinct groups"
        + ("" if len(groups) >= 5 else " -- a grouped split needs more, or it is not really held out"))

    d = cap["hidden_size"]
    add("rows vs dimensions", len(rows) >= 2 * d, len(rows) >= d,
        f"{len(rows)} rows vs {d} dims"
        + ("" if len(rows) >= d else " -- underdetermined: any probe will separate the training set"))

    if cap.get("separation"):
        best = max(range(len(cap["separation"])), key=lambda i: cap["separation"][i])
        v = cap["separation"][best]
        add("class separation present", v >= 0.05, v >= 0.02,
            f"peak ||d-of-means|| / ||h|| = {v:.3f} at slot {best}"
            + ("" if v >= 0.05 else " -- classes barely separated anywhere; a probe is unlikely to work"))

    if cap["kind"] == "rollout":
        lens = [len(it["rows"]) for it in items]
        add("varied rollout lengths", len(set(lens)) > 1, True,
            f"lengths {min(lens)}-{max(lens)}"
            + ("" if len(set(lens)) > 1 else " -- all identical, check the generation actually sampled"))
    return checks


def verdict(checks):
    bad = [c for c in checks if c["status"] == "fail"]
    warn = [c for c in checks if c["status"] == "warn"]
    if bad:
        return "BLOCKED", f"{len(bad)} check(s) failed -- fix before training on this"
    if warn:
        return "WARN", f"{len(warn)} warning(s) -- usable, but read them first"
    return "READY", "all checks passed"


DEMO = ([{"prompt": f"The capital of {c} is {a}.", "label": 1, "group": c,
          "marks": {"subject": c}} for c, a in
         [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"),
          ("Spain", "Madrid"), ("Norway", "Oslo")]]
        + [{"prompt": f"The capital of {c} is {w}.", "label": 0, "group": c,
            "marks": {"subject": c}} for c, w in
           [("France", "Tokyo"), ("Japan", "Cairo"), ("Italy", "Oslo"), ("Egypt", "Lima"),
            ("Spain", "Rome"), ("Norway", "Madrid")]])


def main():
    args = loading.cli(out=dict(default="capture.json"), pool=dict(default="last"),
                       rollouts=dict(action="store_true", help="dump a rollout capture instead"))
    model, tok = loading.load_hf(args.model, args.quant)
    if args.rollouts:
        rs = rollout.sample_rollouts(
            model, tok, [f"Question: What is the capital of {c}?\nAnswer:" for c in
                         ("France", "Australia", "Kazakhstan", "Peru", "Bhutan", "Chile")],
            max_new_tokens=10, n_samples=3, temperature=0.9,
            verifier=lambda t, a: a.lower() in t.lower(),
            answers=["Paris", "Canberra", "Astana", "Lima", "Thimphu", "Santiago"])
        print(rollout.summarize(rs))
        cap = build_rollout_capture(rs, loading.n_layers(model))
    else:
        cap = build_capture(model, tok, DEMO, pool=args.pool)

    with open(args.out, "w") as f:
        json.dump(cap, f)
    print(f"\nwrote {args.out} ({os.path.getsize(args.out)/1024:.0f} kB, "
          f"{len(cap['items'])} items, {sum(len(i['rows']) for i in cap['items'])} rows)\n")
    for c in cap["checks"]:
        print(f"  [{c['status'].upper():>4}] {c['name']:<26} {c['detail']}")
    status, msg = verdict(cap["checks"])
    print(f"\n{status}: {msg}")


if __name__ == "__main__":
    main()
