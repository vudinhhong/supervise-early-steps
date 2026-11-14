r"""G10 -- is it the magnitude of the coverage hole, or where the hole is?

Twenty-one supervision measures trained on the same student with the same objective
are separated perfectly by the coverage of the worst-covered STEP-POSITION bucket,
and not at all by the coverage coefficient over the full 1296-state abstraction: the
two measures that collapse leave a step region with essentially no supervision, while
the nineteen that behave normally keep every region above 0.18.  That separation was
found after the fact, on two collapsing cases.  This stage tests it forward.

The dial.  Supervise every step, then keep only a fraction of the supervised steps
inside ONE chosen step bucket.  Writing d_b for the teacher's share of mass in that
bucket, throttling it at rate r gives a coverage ratio r / (1 - d_b + r*d_b), so the
rate needed for a target coverage m is r = m(1 - d_b) / (1 - m*d_b).  The achieved
coverage is then MEASURED, never assumed.

The prediction, registered before the stage ran.  Starving an EARLY step bucket
should hurt; starving a LATE one at the same coverage should not.  This is what
separates a positional account from a magnitude account: the magnitude account says
only the number matters, so both should hurt equally.

Note on step buckets: a supervised step at index t lands in bucket min(t+1, 5), so
bucket 1 is the first step of every trace and bucket 5 is everything from the fifth
step on.  Bucket 0 is the empty prefix and has no step to supervise.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from scd import corpus, data, hf, runio, stats, train
from scd.abstraction import N_MAG_B, N_OP_B, N_STEP_B
from experiments import _common as C

PER_BUCKET = N_MAG_B * N_OP_B
EARLY_BUCKET, LATE_BUCKET = 1, N_STEP_B - 1


def step_bucket(t: int) -> int:
    return min(t + 1, N_STEP_B - 1)


def teacher_bucket_share(base_rows) -> np.ndarray:
    h = np.zeros(N_STEP_B)
    for c in corpus.teacher_codes(base_rows):
        h[c // PER_BUCKET] += 1
    return h / max(h.sum(), 1)


def throttle(rows, bucket: int, rate: float, seed: int = 0) -> None:
    """Supervise everything, then keep only ``rate`` of the steps in ``bucket``."""
    rng = np.random.default_rng(seed)
    for r in rows:
        m = [True] * len(r.steps)
        for t in range(len(r.steps)):
            if step_bucket(t) == bucket and rng.random() > rate:
                m[t] = False
        r.step_mask = m


def measured_min_cov(rows, d_share) -> tuple:
    h = np.zeros(N_STEP_B)
    for c in corpus.measure_codes(rows):
        h[c // PER_BUCKET] += 1
    p = h / max(h.sum(), 1)
    live = d_share > 0
    ratio = np.divide(p, d_share, out=np.zeros_like(p), where=d_share > 0)
    return float(ratio[live].min()), ratio.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--student", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--objective", default="ce")
    ap.add_argument("--targets", default="0.02 0.05 0.15 0.40")
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
    ap.add_argument("--tag", default="g10_positional")
    ap.add_argument("--runs-tag", default="")
    C.shard_args(ap)
    a = ap.parse_args()

    targets = [float(x) for x in a.targets.split()]
    seeds = [int(s) for s in a.seeds.split()]
    problems, census = C.suite(a.suite_limit)
    T = runio.thresholds()["g10"]

    tpath = C.traces_path(a.teacher, a.prompts, a.samples)
    base_rows = corpus.load(tpath)
    d_share = teacher_bucket_share(base_rows)
    print("[g10] teacher share per step bucket: "
          + "  ".join("b%d=%.4f" % (i, d_share[i]) for i in range(N_STEP_B)))

    # ---- build every cell, then match all of them to the common floor --------
    cells, before = {}, {}
    for bucket, label in ((EARLY_BUCKET, "early"), (LATE_BUCKET, "late")):
        d_b = d_share[bucket]
        for m in targets:
            rate = m * (1 - d_b) / max(1 - m * d_b, 1e-9)
            rate = float(min(max(rate, 0.0), 1.0))
            rows = corpus.load(tpath)
            throttle(rows, bucket, rate, seed=0)
            cells[(label, m)] = rows
            before[(label, m)] = sum(r.n_supervised_steps() for r in rows)
            print("[g10] %-5s target=%.2f -> throttle bucket %d at rate %.4f, "
                  "%d supervised steps" % (label, m, bucket, rate,
                                           before[(label, m)]))
    floor = min(before.values())
    print("[g10] matching every cell down to %d supervised steps "
          "(range %d..%d)" % (floor, min(before.values()), max(before.values())))

    cov = {}
    for k, rows in cells.items():
        got = corpus.match_supervised_count(rows, floor, seed=0)
        mc, ratios = measured_min_cov(rows, d_share)
        cov[k] = {"min_cov_step": mc, "per_bucket": ratios,
                  "c2_full": C.coverage_of(rows, base_rows)["c2"],
                  "steps_after_match": got}
        print("[g10] %-5s target=%.2f: measured min_cov=%.4f  C2_full=%.3f  "
              "steps=%d" % (k[0], k[1], mc, cov[k]["c2_full"], got))

    counts = [v["steps_after_match"] for v in cov.values()]
    matched_ok = max(counts) - min(counts) <= 0.01 * max(counts) + 5
    mcs = sorted(v["min_cov_step"] for v in cov.values())
    span = mcs[-1] / max(mcs[0], 1e-9)
    print("\n[g10] V10.1 dial span = %.1fx (needs >= %.1f) -> %s"
          % (span, T["min_cov_span_min_ratio"],
             "ok" if span >= T["min_cov_span_min_ratio"] else "TOO NARROW"))
    print("[g10] V10.2 counts matched: %s (%d..%d)"
          % (matched_ok, min(counts), max(counts)))

    # Which arms belong to this worker.  Enumerated over every cell, because the
    # supervision floor is the minimum over all of them and each arm is matched down
    # to it; sharding decides what to train, never what to match to.
    order = sorted(cov, key=lambda x: (x[0], x[1]))
    arms = C.shard([(k, int(sd)) for k in order for sd in seeds],
                   a.nshards, a.shard)
    if a.list_arms:
        print("[g10] shard %d/%d: %d arm(s), floor %d"
              % (a.shard, a.nshards, len(arms), floor))
        for i, (k, sd) in enumerate(arms):
            print("  %2d  where=%-5s target=%.2f seed=%d  C2=%9.6f"
                  % (i, k[0], k[1], sd, cov[k]["c2_full"]))
        return

    if a.base_accuracy is not None:
        base = {"accuracy": float(a.base_accuracy)}
        print("[g10] base %.2f%% (given, not measured)" % (100 * base["accuracy"]))
    else:
        m0, tok = C.student(a.student, a.lora_r)
        base = C.score(m0, tok, problems, a.eval_batch)
        hf.free(m0)
        print("[g10] base %.2f%%" % (100 * base["accuracy"]))

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
                out.add((r["where"], float(r["target"]), int(r["seed"])))
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
        print("[g10] resumed %d arms from %s" % (len(runs), ledger.name))

    def checkpoint(partial: bool = True) -> None:
        runio.save(a.tag, {
            "args": vars(a), "census": census, "suite_n": len(problems),
            "suite_mde_points": data.mde(len(problems)),
            "base_accuracy": base["accuracy"], "matched_count": floor,
            "teacher_bucket_share": d_share.tolist(),
            "dial_span_ratio": span, "counts_matched": bool(matched_ok),
            "coverage": {"%s|%.2f" % k: v for k, v in cov.items()},
            "runs": runs, "oom_arms": oom, "partial": partial})

    checkpoint()
    splits: dict = {}
    for k, seed in arms:
        label, m = k
        if k not in splits:
            splits[k] = C.probe_and_fit_rows(cells[k])
        tr, probe, fit = splits[k]
        if (label, m, int(seed)) in done_arms():
            print("[g10] skip %s %.2f s%d (done)" % (label, m, seed))
            continue

        def arm(seed=seed, tr=tr, probe=probe, fit=fit):
            # Seed the adapter too, not only the data order; see C.student.
            model, tk = C.student(a.student, a.lora_r, seed=seed)
            cfg = train.TrainConfig(objective=a.objective, lr=a.lr,
                                    epochs=a.epochs, batch=a.batch,
                                    accum=a.accum, lora_r=a.lora_r,
                                    seed=seed)
            info = train.train(model, tk, tr, cfg, probe_rows=probe,
                               fit_rows=fit)
            sc = C.score(model, tk, problems, a.eval_batch)
            hf.free(model)
            return info, sc

        got, err = C.oom_safe("g10 %s %.2f s%d" % (label, m, seed), arm)
        if got is None:
            oom.append({"where": label, "target": m, "seed": seed,
                        "error": err})
            continue
        info, sc = got
        runs.append({"where": label, "target": m, "seed": seed,
                     "min_cov_step": cov[k]["min_cov_step"],
                     "c2_full": cov[k]["c2_full"],
                     "accuracy": sc["accuracy"],
                     "gain_points": 100 * (sc["accuracy"] - base["accuracy"]),
                     "n_supervised_positions": info["n_supervised_positions"]})
        C.append_run(runs_tag, runs[-1])
        checkpoint()
        print("[g10] %-5s target=%.2f min_cov=%.4f s%d: %.2f%% (gain %+.2f)"
              % (label, m, cov[k]["min_cov_step"], seed,
                 100 * sc["accuracy"], runs[-1]["gain_points"]))

    # ---- the registered comparison ------------------------------------------
    def cell_mean(label, m):
        v = [100 * r["accuracy"] for r in runs
             if r["where"] == label and abs(r["target"] - m) < 1e-9]
        return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

    tightest = min(targets)
    e_acc, e_n = cell_mean("early", tightest)
    l_acc, l_n = cell_mean("late", tightest)
    positional_gap = l_acc - e_acc
    g10_1 = positional_gap >= T["positional_gap_min_points"]

    e_series = [(cov[("early", m)]["min_cov_step"], cell_mean("early", m)[0])
                for m in sorted(targets)]
    ok = [x for x in e_series if x[1] == x[1]]
    rho = (stats.spearman([x[0] for x in ok], [x[1] for x in ok])
           if len(ok) >= 3 else {"rho": float("nan"), "p": float("nan")})
    g10_2 = rho["rho"] >= T["early_series_rho_min"]

    runio.save(a.tag, {
        "args": vars(a), "census": census, "suite_n": len(problems),
        "suite_mde_points": data.mde(len(problems)),
        "base_accuracy": base["accuracy"], "matched_count": floor,
        "teacher_bucket_share": d_share.tolist(),
        "dial_span_ratio": span, "counts_matched": bool(matched_ok),
        "coverage": {"%s|%.2f" % k: v for k, v in cov.items()},
        "runs": runs, "oom_arms": oom, "partial": False,
        "cells": {"%s|%.2f" % (lb, m): cell_mean(lb, m)[0]
                  for lb in ("early", "late") for m in targets},
        "positional_gap_points": positional_gap,
        "g10_1_positional": bool(g10_1),
        "early_series": e_series, "early_series_rho": rho,
        "g10_2_monotone": bool(g10_2)})

    print("\n--- G10 summary ----------------------------------------------")
    for lb in ("early", "late"):
        for m in sorted(targets):
            acc, n = cell_mean(lb, m)
            print("  %-5s target=%.2f  min_cov=%.4f  acc=%6.2f%%  (n=%d)"
                  % (lb, m, cov[(lb, m)]["min_cov_step"], acc, n))
    print("  G10.1 positional gap at tightest coverage = %+.2f points "
          "(needs >= %.1f) -> %s"
          % (positional_gap, T["positional_gap_min_points"],
             "YES" if g10_1 else "no"))
    print("  G10.2 early series rises with coverage: rho=%+.3f (needs >= %.2f)"
          " -> %s" % (rho["rho"], T["early_series_rho_min"],
                      "YES" if g10_2 else "no"))


if __name__ == "__main__":
    main()
