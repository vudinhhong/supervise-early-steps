r"""G9 -- is the coverage effect a threshold, or a gradual decline?

G3 measured C2 at 1.02, 5.36, 14.47 and 39.15.  G7's recomputed coverage added
points at 6.10 and 8.35.  Pooling the 1.5B/ce arms of both, accuracy is flat to
within the evaluation suite's MDE for every C2 <= 8.35, then 43.87 at 14.47 and
8.19 at 39.15.  Nothing at all is measured between 8.35 and 14.47 -- which is
precisely where the shape of the relationship is decided.

This stage fills that gap with seven levels between roughly 5 and 20.  Both
outcomes are reportable and the discriminator is registered in thresholds.json
under ``g9``: the largest drop between adjacent levels divided by the mean of the
remaining drops, read as a threshold at >= 3.0 and as a gradual decline below it.

Two design points carry the stage.

The dial.  ``corpus.mask_fraction``'s ``cell_drop`` mode bans a random share of
abstraction cells, so C2 depends on WHICH cells are dropped as much as on how
many -- it is not monotone in the fraction.  That is a nuisance turned into an
instrument: a grid over (fraction, drop-seed) yields a dense ladder of C2 values,
measured before any training, from which the levels are chosen.  Choosing on a
pre-measured property of the supervision is not choosing on the outcome.

The confound.  Across that grid, C2 and the number of supervised positions are
strongly anti-correlated.  Left alone the sweep would move coverage and
supervision quantity together, and the result would be uninterpretable in exactly
the way G3 took pains to avoid.  Every arm is matched down to the common floor of
the selected set, and C2 is re-measured after matching.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from scd import corpus, data, hf, runio, stats, train
from experiments import _common as C

# (fraction, drop-seed) pairs, chosen from a calibration grid whose C2 was
# measured before any training.  Two anchors bracket the band of interest.
LEVELS = "0.90:1 0.80:1 0.75:2 0.80:2 0.85:2 0.70:1 0.65:1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--student", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--objective", default="ce")
    ap.add_argument("--levels", default=LEVELS,
                    help="space-separated frac:dropseed pairs")
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
    ap.add_argument("--tag", default="g9_threshold")
    ap.add_argument("--runs-tag", default="")
    C.shard_args(ap)
    a = ap.parse_args()

    levels = [(float(x.split(":")[0]), int(x.split(":")[1]))
              for x in a.levels.split()]
    seeds = [int(s) for s in a.seeds.split()]
    problems, census = C.suite(a.suite_limit)
    T = runio.thresholds()["g9"]

    tpath = C.traces_path(a.teacher, a.prompts, a.samples)
    base_rows = corpus.load(tpath)
    print(f"[g9] teacher occupancy from {tpath.name}: {len(base_rows)} rows")

    # ---- build every level, then match them all to the common floor -----------
    prepared, before = {}, {}
    for frac, dseed in levels:
        rows = corpus.load(tpath)
        corpus.mask_fraction(rows, frac, "cell_drop", seed=dseed)
        prepared[(frac, dseed)] = rows
        before[(frac, dseed)] = sum(r.n_supervised_steps() for r in rows)
    floor = min(before.values())
    print(f"[g9] matching every level down to {floor} supervised positions "
          f"(was {min(before.values())}..{max(before.values())})")

    cov = {}
    for key, rows in prepared.items():
        got = corpus.match_supervised_count(rows, floor, seed=0)
        cov[key] = C.coverage_of(rows, base_rows)
        cov[key]["n_cells"] = len(set(corpus.measure_codes(rows)))
        cov[key]["positions_before_match"] = before[key]
        cov[key]["positions_after_match"] = got
        print(f"[g9] frac={key[0]:.2f} dseed={key[1]}: C2={cov[key]['c2']:7.3f}  "
              f"cells={cov[key]['n_cells']:4d}  positions {before[key]} -> {got}")

    counts = [v["positions_after_match"] for v in cov.values()]
    assert max(counts) - min(counts) <= 0.01 * max(counts) + 5, \
        f"supervised counts not matched: {counts}"

    c2s = sorted(v["c2"] for v in cov.values())
    span = c2s[-1] / max(c2s[0], 1e-9)
    ladder_ok = len(cov) >= T["min_levels"] and span >= T["c2_span_min_ratio"]
    print(f"\n[g9] ladder: {len(cov)} levels, C2 {c2s[0]:.2f}..{c2s[-1]:.2f} "
          f"(span {span:.2f}x; needs >= {T['c2_span_min_ratio']}x and "
          f">= {T['min_levels']} levels) -> {'ok' if ladder_ok else 'TOO NARROW'}")

    # ---- which arms belong to this worker ------------------------------------
    # Enumerated over the WHOLE ladder, then split: the supervision floor is the
    # minimum over every level, so a worker must still build all of them even when
    # it trains one arm.  Sharding decides what to train, never what to match to.
    order = sorted(cov, key=lambda k: cov[k]["c2"])
    arms = C.shard([(k, int(sd)) for k in order for sd in seeds],
                   a.nshards, a.shard)
    if a.list_arms:
        print(f"[g9] shard {a.shard}/{a.nshards}: {len(arms)} arm(s), "
              f"floor {floor}")
        for i, (k, sd) in enumerate(arms):
            print("  %2d  frac=%.2f dseed=%d seed=%d  C2=%9.6f"
                  % (i, k[0], k[1], sd, cov[k]["c2"]))
        return

    # ---- base accuracy -------------------------------------------------------
    if a.base_accuracy is not None:
        base = {"accuracy": float(a.base_accuracy)}
        print(f"[g9] base {base['accuracy']*100:.2f}% (given, not measured)")
    else:
        m, tok = C.student(a.student, a.lora_r)
        base = C.score(m, tok, problems, a.eval_batch)
        hf.free(m)
        print(f"[g9] base {base['accuracy']*100:.2f}%")

    # ---- the sweep -----------------------------------------------------------
    runs_tag = a.runs_tag or a.tag
    ledger = runio.results_dir() / f"{runs_tag}_runs.jsonl"

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
                out.add((float(r["frac"]), int(r["drop_seed"]), int(r["seed"])))
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
        print(f"[g9] resumed {len(runs)} arms from {ledger.name}")

    def checkpoint(partial: bool = True) -> None:
        runio.save(a.tag, {"args": vars(a), "census": census,
                           "suite_n": len(problems),
                           "suite_mde_points": data.mde(len(problems)),
                           "base_accuracy": base["accuracy"],
                           "matched_count": floor, "ladder_ok": ladder_ok,
                           "c2_span_ratio": span,
                           "coverage": {f"{k[0]}:{k[1]}": v for k, v in cov.items()},
                           "runs": runs, "oom_arms": oom, "partial": partial})

    checkpoint()
    splits: dict = {}
    for key, seed in arms:
        frac, dseed = key
        if key not in splits:
            splits[key] = C.probe_and_fit_rows(prepared[key])
        tr, probe, fit = splits[key]
        if (frac, dseed, int(seed)) in done_arms():
            print(f"[g9] skip frac={frac} dseed={dseed} s{seed} (done)")
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

        got, err = C.oom_safe(f"g9 frac={frac} dseed={dseed} s{seed}", arm)
        if got is None:
            oom.append({"frac": frac, "drop_seed": dseed, "seed": seed,
                        "error": err})
            continue
        info, sc = got
        runs.append({"frac": frac, "drop_seed": dseed, "seed": seed,
                     "c2": cov[key]["c2"], "n_cells": cov[key]["n_cells"],
                     "accuracy": sc["accuracy"],
                     "gain_points": 100 * (sc["accuracy"] - base["accuracy"]),
                     "n_supervised_positions": info["n_supervised_positions"],
                     "own_supervision_loss_drop":
                         info["own_supervision_loss_drop"]})
        C.append_run(runs_tag, runs[-1])
        checkpoint()
        print(f"[g9] C2={cov[key]['c2']:7.3f} s{seed}: "
              f"{sc['accuracy']*100:.2f}%  (gain "
              f"{runs[-1]['gain_points']:+.2f})")

    # ---- threshold or gradual? ----------------------------------------------
    cells = {}
    for r in runs:
        cells.setdefault(round(float(r["c2"]), 3), []).append(r["accuracy"])
    ladder = sorted((c2, 100 * float(np.mean(v)), len(v))
                    for c2, v in cells.items())

    drops = [(ladder[i][0], ladder[i + 1][0], ladder[i][1] - ladder[i + 1][1])
             for i in range(len(ladder) - 1)]
    cliff_ratio = float("nan")
    verdict = "indeterminate"
    if len(drops) >= 3:
        mags = sorted((abs(d[2]) for d in drops), reverse=True)
        rest = float(np.mean(mags[1:])) if len(mags) > 1 else float("nan")
        cliff_ratio = mags[0] / rest if rest > 1e-9 else float("inf")
        verdict = ("threshold" if cliff_ratio >= T["cliff_ratio_min"]
                   else "gradual")

    biggest = max(drops, key=lambda d: abs(d[2])) if drops else None
    anchor_ok = (ladder and abs(ladder[0][1] - 65.81) <= T["anchor_within_points"])

    runio.save(a.tag, {
        "args": vars(a), "census": census, "suite_n": len(problems),
        "suite_mde_points": data.mde(len(problems)),
        "base_accuracy": base["accuracy"], "matched_count": floor,
        "ladder_ok": ladder_ok, "c2_span_ratio": span,
        "anchor_ok": bool(anchor_ok),
        "coverage": {f"{k[0]}:{k[1]}": v for k, v in cov.items()},
        "runs": runs, "oom_arms": oom, "partial": False,
        "ladder": [{"c2": c, "accuracy_points": a_, "n_seeds": n}
                   for c, a_, n in ladder],
        "adjacent_drops": [{"from_c2": x, "to_c2": y, "drop_points": d}
                           for x, y, d in drops],
        "cliff_ratio": cliff_ratio, "verdict": verdict,
        "threshold_between": ([biggest[0], biggest[1]] if biggest else None),
        "rank_c2_vs_accuracy": stats.spearman([c for c, _, _ in ladder],
                                              [a_ for _, a_, _ in ladder]),
    })

    print("\n--- G9 summary ------------------------------------------------")
    for c, a_, n in ladder:
        print(f"  C2={c:7.3f}  acc={a_:6.2f}%  (n={n})")
    print(f"  cliff ratio = {cliff_ratio:.2f} "
          f"(threshold at >= {T['cliff_ratio_min']}) -> {verdict.upper()}")
    if biggest:
        print(f"  largest drop: {biggest[2]:+.2f} points between C2={biggest[0]:.3f} "
              f"and C2={biggest[1]:.3f}")


if __name__ == "__main__":
    main()
