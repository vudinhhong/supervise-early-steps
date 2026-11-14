r"""G2 -- is the coverage estimate valid on real text, or is it reading style?

The estimator that reads a student's hidden state has a fatal alternative
explanation available: it can separate teacher prefixes from supervision prefixes
by recognising *who wrote them* rather than *which reasoning state they reached*.
The earlier compute-matched study was killed by exactly this class of leak -- a
per-token demand signal at AUC 0.82 whose token-log-frequency control already
reached 0.61.

Four tests, with three shortcut controls run on every one:

  W1  null calibration: two disjoint halves of the same teacher traces.  AUC must
      be 0.50 and C2 must be 1.  Anything else is overfitting, full stop.
  W2  sensitivity: teacher traces against the same traces with tokens shuffled
      inside each step.  AUC must be high or the features carry nothing.
  W3  paraphrase invariance -- the decisive test.  Teacher traces against traces
      rewritten to different words with the same numeric sequence.  AUC must be
      LOW.  Pairs whose numeric sequence changed are discarded, and the discard
      rate is reported (V3.1).
  W4  source invariance: teacher traces against *student* traces that reach the
      same numeric sequence.  Needs enough matched pairs to mean anything (V3.2).

The abstraction-based estimator passes W3 and W4 by construction, so its role here
is to provide the reference: if the hidden-state estimator disagrees with it in
rank, the hidden-state estimator is measuring something else.
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Sequence

import numpy as np

from scd import (abstraction, corpus, coverage, hf, runio, sampling, stats,
                 tasks)
from scd.abstraction import N_STATES
from experiments import _common as C

REWRITE = ("Rewrite the solution below in different words. Keep every number and "
           "every step in the same order. Do not add or remove steps.\n\nSolution:\n")


def numeric_signature(steps: Sequence[str]) -> tuple:
    return tuple(round(v, 6) for s in steps for v in abstraction.numbers(s))


def _features(model, tok, rows: Sequence[corpus.Row], micro_batch: int = 8,
              limit: int = 4000) -> np.ndarray:
    recs = [{"question": r.question, "steps": r.steps} for r in rows[:limit]]
    H = sampling.hidden_at_prefixes(model, tok, recs, micro_batch=micro_batch)
    return np.concatenate([h[1:] for h in H if h.shape[0] > 1], 0)


def _codes(rows: Sequence[corpus.Row], limit: int = 4000) -> List[int]:
    return [c for r in rows[:limit] for c in r.sigma()[1:]]


def _controls(rows_a, rows_b, tok) -> Dict[str, float]:
    """Shortcut classifiers: length, token rarity, relative position.

    Reported for every test.  If the hidden-state AUC does not exceed the best of
    these by a margin, the signal is surface statistics (criterion G2.5).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    def feats(rows):
        out = []
        for r in rows:
            path = r.sigma()
            for t, st in enumerate(r.steps):
                ids = tok(st, add_special_tokens=False)["input_ids"]
                out.append([len(ids),
                            float(np.mean(ids)) / 1e5,
                            (t + 1) / max(len(r.steps), 1)])
        return np.array(out, float)
    A, B = feats(rows_a), feats(rows_b)
    n = min(len(A), len(B))
    if n < 50:
        return {"length": float("nan"), "rarity": float("nan"), "position": float("nan")}
    X = np.concatenate([A[:n], B[:n]]); y = np.concatenate([np.ones(n), np.zeros(n)])
    out = {}
    for j, name in enumerate(("length", "rarity", "position")):
        clf = LogisticRegression(max_iter=1000).fit(X[:, [j]], y)
        out[name] = float(roc_auc_score(y, clf.predict_proba(X[:, [j]])[:, 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--prompts", type=int, default=2500)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--hidden-batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--mag-bins", default="12 24 46")
    ap.add_argument("--tag", default="g2_validity")
    a = ap.parse_args()

    trows = C.build_traces(a.teacher, a.prompts, a.samples, gen_batch=a.gen_batch)
    half = len(trows) // 2
    A, B = trows[:half], trows[half:2 * half]

    # ---- W2: shuffle tokens inside each step, keeping the step count
    rng = np.random.default_rng(0)
    def shuffled(rows):
        out = []
        for r in rows:
            st = []
            for s in r.steps:
                w = s.split()
                rng.shuffle(w)
                st.append(" ".join(w))
            out.append(corpus.Row(r.question, r.gold, st, r.correct, r.source,
                                  "shuffled"))
        return out
    A_shuf = shuffled(A)

    # ---- W3: paraphrase with the teacher, keep only numeric-identical rewrites
    tok_t = hf.load_tokenizer(a.teacher)
    teacher = hf.load_model(a.teacher)
    prompts = [tasks.chat_prefix(tok_t, REWRITE + "\n".join(r.steps)) for r in A]
    texts = sampling.generate(teacher, tok_t, prompts, max_new=384,
                              micro_batch=a.gen_batch, temperature=0.7,
                              progress_every=256)
    # ---- W4: student traces, matched on the numeric signature
    hf.free(teacher)
    st_model = hf.load_model(a.student)
    tok_s = hf.load_tokenizer(a.student)
    srecs = sampling.sample_traces(
        st_model, tok_s,
        [tasks.Problem(r.question, r.gold, r.source) for r in trows[:a.prompts]],
        n_samples=2, max_new=384, micro_batch=a.gen_batch, seed=1)
    hf.free(st_model)

    A_para, n_drop = [], 0
    for r, t in zip(A, texts):
        st = [s for s in t.strip().split("\n") if s.strip()]
        if st and numeric_signature(st) == numeric_signature(r.steps):
            A_para.append(corpus.Row(r.question, r.gold, st, r.correct, r.source,
                                     "paraphrase"))
        else:
            n_drop += 1
    sig_t = {}
    for r in trows:
        sig_t.setdefault((r.question, numeric_signature(r.steps)), r)
    A_stud = [corpus.Row(s["question"], s["gold"], s["steps"], s["correct"],
                         s.get("source", ""), "student")
              for s in srecs
              if (s["question"], numeric_signature(s["steps"])) in sig_t]
    A_stud_any = [corpus.Row(s["question"], s["gold"], s["steps"], s["correct"],
                             s.get("source", ""), "student") for s in srecs]

    # ---- features from the student's hidden states
    model, tok = C.student(a.student, lora_r=0)
    F = {k: _features(model, tok, v, a.hidden_batch, a.limit) for k, v in
         {"A": A, "B": B, "A_shuf": A_shuf, "A_para": A_para,
          "A_stud": A_stud, "A_stud_any": A_stud_any}.items() if v}
    hf.free(model)

    tests = {"W1_null": ("A", "B", A, B),
             "W2_shuffle": ("A", "A_shuf", A, A_shuf),
             "W3_paraphrase": ("A", "A_para", A, A_para),
             "W4_source": ("A", "A_stud", A, A_stud),
             "W5_real_shift": ("A", "A_stud_any", A, A_stud_any)}
    out = {}
    for name, (ka, kb, ra, rb) in tests.items():
        if kb not in F or len(F[kb]) < 200 or len(rb) < 20:
            out[name] = {"note": "too few matched rows", "n_rows_b": len(rb)}
            continue
        B_est = coverage.c2_density_ratio(F[ka], F[kb])
        A_est = coverage.c2_counting(_codes(ra, a.limit), _codes(rb, a.limit), N_STATES)
        Cc = coverage.c2_knn(F[ka], F[kb])
        ctrl = _controls(ra, rb, tok)
        best_ctrl = max(v for v in ctrl.values() if np.isfinite(v))
        out[name] = {"auc_B": B_est["auc"], "c2_B": B_est["c2"],
                     "c2_A": A_est["c2"], "c2_C": Cc["c2"],
                     "controls": ctrl, "excess_auc": B_est["auc"] - best_ctrl,
                     "n_rows_b": len(rb)}
        print(f"[g2] {name:14s} AUC_B={B_est['auc']:.3f} "
              f"excess={out[name]['excess_auc']:+.3f} "
              f"C2: A={A_est['c2']:.2f} B={B_est['c2']:.2f} C={Cc['c2']:.2f}")

    # ---- W1 null calibration as a function of sample size (G2.1 diagnostic)
    #
    # The classifier estimator is only unbiased when the rows outnumber the
    # features by enough.  At smoke scale -- 200 prefixes, 896-dimensional hidden
    # states reduced to 64 -- W1 reported AUC 0.921 and C2 = 35.3 on two halves of
    # one distribution, which is memorisation, not coverage.  A single-point check
    # cannot tell "this estimator is broken" from "this estimator is broken at this
    # n", and those call for different responses: abandon estimator B, or collect
    # more prefixes.  So the null is measured along n.
    null_curve = {}
    if len(F["A"]) >= 200 and len(F["B"]) >= 200:
        avail = min(len(F["A"]), len(F["B"]))
        for frac in (0.1, 0.25, 0.5, 1.0):
            n = int(avail * frac)
            if n < 100:
                continue
            e = coverage.c2_density_ratio(F["A"][:n], F["B"][:n])
            null_curve[str(n)] = {"auc": e["auc"], "c2": e["c2"]}
        print("[g2] W1 null vs sample size: " + ", ".join(
            f"n={k}: AUC={v['auc']:.3f}" for k, v in null_curve.items()))

    # ---- estimator agreement across the constructed pairs
    keys = [k for k, v in out.items() if "c2_A" in v]
    pair = {}
    for x, y in (("c2_A", "c2_B"), ("c2_A", "c2_C"), ("c2_B", "c2_C")):
        pair[f"{x}~{y}"] = stats.spearman([out[k][x] for k in keys],
                                          [out[k][y] for k in keys])

    # ---- abstraction sensitivity (G2.7): does C2 survive re-binning?
    bins = {}
    code_cache = {k: (_codes(tests[k][2], a.limit), _codes(tests[k][3], a.limit))
                  for k in keys}
    for nb in [int(x) for x in a.mag_bins.split()]:
        vals = []
        for k in keys:
            ca, cb = code_cache[k]
            ra, ns = abstraction.recode(ca, nb)
            rb_, _ = abstraction.recode(cb, nb)
            # total_alpha, not the per-state default: this loop is the one
            # comparison in the study that spans different state-space sizes.
            vals.append(coverage.c2_counting(
                ra, rb_, ns, total_alpha=0.5 * N_STATES)["c2"])
        bins[str(nb)] = vals
    # ddof=1 is undefined for a single setting; one bin setting is not a
    # sensitivity test and must report that rather than a NaN.
    n_settings = len(bins)
    cvs = ([float(np.std([bins[b][i] for b in bins], ddof=1)
                  / max(abs(np.mean([bins[b][i] for b in bins])), 1e-9))
            for i in range(len(keys))] if n_settings >= 2 else [])

    runio.save(a.tag, {"args": vars(a), "tests": out, "pairwise": pair,
                       "w1_null_vs_n": null_curve,
                       "paraphrase_discard_rate": n_drop / max(len(A), 1),
                       "n_paraphrase_kept": len(A_para),
                       "n_student_matched": len(A_stud),
                       "abstraction_cv": {
                           "per_test": cvs, "settings": sorted(bins),
                           "values": bins,
                           "median": float(np.median(cvs)) if cvs
                           else float("nan")},
                       "n_teacher_rows": len(trows)})


if __name__ == "__main__":
    main()
