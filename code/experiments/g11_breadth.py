r"""G11 -- does the collapse need a DEEP hole in one early step bucket, or a BROAD one?

Background, stated plainly because it is an error of ours.  A post-hoc statistic
separated the twenty-one configurations trained so far: min_cov_step, the coverage of
the worst step bucket.  G10 tested it forward and found accuracy flat from min_cov
0.019 to 0.154.  The reason is that a minimum is blind to how many buckets are
starved, and the datasets differ on exactly that: G3's `late` mask leaves bucket 1 at
0.000 and bucket 2 at 0.035 -- two early buckets -- while G10 throttles bucket 1
alone, because step_bucket(t)=min(t+1,5) makes bucket 1 the first step only.

The per-bucket vectors also rule out starved MASS: the `early` mask leaves bucket 5
at 0.182, which is 65% of the teacher's occupancy covered at 18%, and loses nothing.
A thin late region is harmless; a thin early region is not.  So the feature is not
symmetric across step position, and neither a min nor a sum can express it.

Two readings survive, and three cells separate them:

  b1 @ 0.001        one early bucket, ten times deeper than G10's tightest
  b1,b2 @ 0.02      two early buckets at a depth G10 showed is harmless alone
  b1..b4 @ 0.02     four early buckets at that same depth

DEPTH wins if only the first collapses.  BREADTH wins if the second or third does
while the first does not.  If none collapses, the collapse needs both, and the
hypothesis narrows again.  All four outcomes are reportable.

Every cell is matched to the same 67637-step floor as G10, so the numbers compare
directly with that stage and with the reference cell early/0.02 at 66.60%.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from scd import corpus, data, hf, runio, train
from scd.abstraction import N_MAG_B, N_OP_B, N_STEP_B
from experiments import _common as C

PER_BUCKET = N_MAG_B * N_OP_B

# label: (buckets to throttle, target coverage ratio for each)
CELLS = {
    "b1_deep":      ((1,),          0.001),
    "b1b2":         ((1, 2),        0.02),
    "b1b2b3b4":     ((1, 2, 3, 4),  0.02),
}


def step_bucket(t: int) -> int:
    return min(t + 1, N_STEP_B - 1)


def bucket_hist(codes):
    h = np.zeros(N_STEP_B)
    for c in codes:
        h[c // PER_BUCKET] += 1
    return h


def throttle(rows, buckets, rate, seed: int = 0) -> None:
    """Supervise everything, then keep only ``rate`` of the steps in ``buckets``."""
    rng = np.random.default_rng(seed)
    bs = set(buckets)
    for r in rows:
        m = [True] * len(r.steps)
        for t in range(len(r.steps)):
            if step_bucket(t) in bs and rng.random() > rate:
                m[t] = False
        r.step_mask = m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--student", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--objective", default="ce")
    ap.add_argument("--prompts", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--suite-limit", type=int, default=0)
    ap.add_argument("--floor", type=int, default=0,
                    help="supervised-step floor to match every cell to.  0 means "
                         "take it from thresholds.json (g11.floor_steps), which is "
                         "G10's floor -- matching it is what makes the two stages "
                         "comparable.")
    ap.add_argument("--tag", default="g11_breadth")
    ap.add_argument("--runs-tag", default="")
    C.shard_args(ap)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split()]
    problems, census = C.suite(a.suite_limit)
    T = runio.thresholds()["g11"]
    floor = a.floor or int(T["floor_steps"])

    tpath = C.traces_path(a.teacher, a.prompts, a.samples)
    base_rows = corpus.load(tpath)
    d_share = bucket_hist(corpus.teacher_codes(base_rows))
    d_share = d_share / d_share.sum()
    print("[g11] teacher mass per step bucket: "
          + "  ".join("b%d=%.4f" % (i, d_share[i]) for i in range(N_STEP_B)))
    print("[g11] matching every cell to %d supervised steps "
          "(same floor as G10)" % floor)

    # ---- build the cells, measure what we actually produced --------------------
    cells, cov = {}, {}
    for label, (buckets, target) in CELLS.items():
        # coverage ratio r for a throttled bucket b, at rate s, is
        # s / (1 - M + s*M) where M is the total mass of the throttled buckets;
        # invert for the rate that hits the target.
        M = float(sum(d_share[b] for b in buckets))
        rate = target * (1 - M) / max(1 - target * M, 1e-9)
        rate = float(min(max(rate, 0.0), 1.0))
        rows = corpus.load(tpath)
        throttle(rows, buckets, rate, seed=0)
        have = sum(r.n_supervised_steps() for r in rows)
        if have < floor:
            raise RuntimeError(
                "cell %s has only %d supervised steps, below the %d floor; the "
                "cells cannot be matched" % (label, have, floor))
        got = corpus.match_supervised_count(rows, floor, seed=0)
        p = bucket_hist(corpus.measure_codes(rows))
        p = p / max(p.sum(), 1)
        ratio = np.divide(p, d_share, out=np.zeros_like(p), where=d_share > 0)
        starved = float(d_share[(d_share > 0) & (ratio < 0.01)].sum())
        cells[label] = rows
        cov[label] = {"buckets": list(buckets), "target": target, "rate": rate,
                      "mass_throttled": M, "per_bucket": ratio.tolist(),
                      "starved_mass_below_0.01": starved,
                      "steps_before_match": have, "steps_after_match": got,
                      "c2_full": C.coverage_of(rows, base_rows)["c2"]}
        print("[g11] %-10s buckets=%-12s rate=%.5f -> " % (
                  label, str(list(buckets)), rate)
              + " ".join("b%d=%.3f" % (i, ratio[i]) for i in range(1, N_STEP_B))
              + "  starved<.01=%.4f  C2=%.2f  steps %d->%d"
              % (starved, cov[label]["c2_full"], have, got))

    counts = [v["steps_after_match"] for v in cov.values()]
    assert max(counts) - min(counts) <= 0.01 * max(counts) + 5, \
        "cells not matched: %s" % counts

    # Which arms belong to this worker; see the note in g10 about the floor.
    arms = C.shard([(lab, int(sd)) for lab in CELLS for sd in seeds],
                   a.nshards, a.shard)
    if a.list_arms:
        print("[g11] shard %d/%d: %d arm(s), floor %d"
              % (a.shard, a.nshards, len(arms), floor))
        for i, (lab, sd) in enumerate(arms):
            print("  %2d  cell=%-10s seed=%d" % (i, lab, sd))
        return

    if a.base_accuracy is not None:
        base = {"accuracy": float(a.base_accuracy)}
        print("[g11] base %.2f%% (given, not measured)" % (100 * base["accuracy"]))
    else:
        m0, tok = C.student(a.student, a.lora_r)
        base = C.score(m0, tok, problems, a.eval_batch)
        hf.free(m0)
        print("[g11] base %.2f%%" % (100 * base["accuracy"]))

    runs_tag = a.runs_tag or a.tag
    ledger = runio.results_dir() / ("%s_runs.jsonl" % runs_tag)

    def done_arms() -> set:
        out = set()
        if not ledger.exists():
            return out
        for line in ledger.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out.add((r["cell"], int(r["seed"])))
            except Exception:
                continue
        return out

    runs, oom = [], []
    for line in (ledger.read_text().splitlines() if ledger.exists() else []):
        line = line.strip()
        if line:
            try:
                runs.append(json.loads(line))
            except Exception:
                pass
    if runs:
        print("[g11] resumed %d arms from %s" % (len(runs), ledger.name))

    def checkpoint(partial: bool = True) -> None:
        runio.save(a.tag, {"args": vars(a), "census": census,
                           "suite_n": len(problems),
                           "suite_mde_points": data.mde(len(problems)),
                           "base_accuracy": base["accuracy"],
                           "matched_count": floor,
                           "teacher_bucket_share": d_share.tolist(),
                           "coverage": cov, "runs": runs, "oom_arms": oom,
                           "partial": partial})

    checkpoint()
    splits: dict = {}
    for label, seed in arms:
        if label not in splits:
            splits[label] = C.probe_and_fit_rows(cells[label])
        tr, probe, fit = splits[label]
        if (label, int(seed)) in done_arms():
            print("[g11] skip %s s%d (done)" % (label, seed))
            continue

        def arm(seed=seed, tr=tr, probe=probe, fit=fit):
            # Seed the adapter too, not only the data order; see C.student.
            model, tk = C.student(a.student, a.lora_r, seed=seed)
            cfg = train.TrainConfig(objective=a.objective, lr=a.lr,
                                    epochs=a.epochs, batch=a.batch,
                                    accum=a.accum, lora_r=a.lora_r, seed=seed)
            info = train.train(model, tk, tr, cfg, probe_rows=probe,
                               fit_rows=fit)
            sc = C.score(model, tk, problems, a.eval_batch)
            hf.free(model)
            return info, sc

        got, err = C.oom_safe("g11 %s s%d" % (label, seed), arm)
        if got is None:
            oom.append({"cell": label, "seed": seed, "error": err})
            continue
        info, sc = got
        runs.append({"cell": label, "seed": seed,
                     "buckets": cov[label]["buckets"],
                     "target": cov[label]["target"],
                     "starved_mass": cov[label]["starved_mass_below_0.01"],
                     "c2_full": cov[label]["c2_full"],
                     "accuracy": sc["accuracy"],
                     "gain_points": 100 * (sc["accuracy"] - base["accuracy"]),
                     "n_supervised_positions": info["n_supervised_positions"]})
        C.append_run(runs_tag, runs[-1])
        checkpoint()
        print("[g11] %-10s s%d: %.2f%%  (gain %+.2f)"
              % (label, seed, 100 * sc["accuracy"], runs[-1]["gain_points"]))

    # ---- the registered reading ---------------------------------------------
    REF = 66.60          # G10 early/0.02, one early bucket at 0.019, flat
    bar = float(T["collapse_drop_min_points"])
    verdict = {}
    for label in CELLS:
        v = [100 * r["accuracy"] for r in runs if r["cell"] == label]
        acc = float(np.mean(v)) if v else float("nan")
        verdict[label] = {"accuracy": acc, "n": len(v),
                          "drop_vs_reference": REF - acc,
                          "collapsed": bool(acc == acc and REF - acc >= bar)}
    coll = [k for k, v in verdict.items() if v["collapsed"]]
    if coll == ["b1_deep"]:
        reading = "depth"
    elif coll and "b1_deep" not in coll:
        reading = "breadth"
    elif not coll:
        reading = "neither alone"
    else:
        reading = "both"

    runio.save(a.tag, {"args": vars(a), "census": census,
                       "suite_n": len(problems),
                       "suite_mde_points": data.mde(len(problems)),
                       "base_accuracy": base["accuracy"], "matched_count": floor,
                       "teacher_bucket_share": d_share.tolist(),
                       "coverage": cov, "runs": runs, "oom_arms": oom,
                       "partial": False, "reference_accuracy": REF,
                       "verdict": verdict, "reading": reading})

    print("\n--- G11 summary ----------------------------------------------")
    for label in CELLS:
        v = verdict[label]
        print("  %-10s acc=%6.2f%% (n=%d)  drop vs %.2f = %+6.2f  %s"
              % (label, v["accuracy"], v["n"], REF, -v["drop_vs_reference"],
                 "COLLAPSED" if v["collapsed"] else "flat"))
    print("  reading: %s" % reading.upper())


if __name__ == "__main__":
    main()
