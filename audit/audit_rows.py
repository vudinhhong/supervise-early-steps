#!/usr/bin/env python
"""How many distinct training examples does each supervision measure actually give?

train.py keeps only rows with at least one supervised token:

    built = [b for b in (batching.build(...) for r in rows)
             if b is not None and b.n_supervised > 0]

That filter is not recorded anywhere, and it is mask-dependent: a mask that bans
most abstraction cells leaves many traces with nothing supervised, so those traces
never enter training at all.  The supervised STEP count was matched across masks;
the number of distinct PROBLEMS was not, and nobody checked it.

If the collapsing masks train on far fewer distinct problems, the G3 effect is at
least partly a data-diversity effect rather than a coverage effect, and the paper's
attribution is wrong.

Run from the repository root:  PYTHONPATH=. python figs/audit_rows.py
"""
import json, pathlib
import numpy as np
from scd import corpus
from experiments import _common as C

TEACHER = "Qwen/Qwen2.5-7B-Instruct"
S15 = "Qwen/Qwen2.5-1.5B"


def main():
    tp = C.traces_path(TEACHER, 6000, 4)
    g3 = json.load(open("results/g3_measure_vs_loss.json"))
    floor = g3["matched_count"]

    def acc(mode, obj):
        v = [100 * r["accuracy"] for r in g3["runs"]
             if r["student"] == S15 and r["objective"] == obj and r["mode"] == mode]
        return float(np.mean(v))

    print("%-14s %9s %9s %10s %9s %9s %9s" % (
        "mask", "rows>=1", "share", "steps", "steps/row", "acc ce", "acc fkl"))
    print("-" * 76)
    rowsurv = {}
    for m in ("uniform", "early", "late", "cluster_drop"):
        rs = corpus.load(tp)
        corpus.mask_fraction(rs, 0.25, m, seed=0)
        corpus.match_supervised_count(rs, floor, seed=0)
        live = [r for r in rs if r.n_supervised_steps() > 0]
        tot = sum(r.n_supervised_steps() for r in rs)
        rowsurv[m] = len(live)
        print("%-14s %9d %8.1f%% %10d %9.2f %9.2f %9.2f" % (
            m, len(live), 100 * len(live) / len(rs), tot,
            tot / max(len(live), 1), acc(m, "ce"), acc(m, "fkl")))

    print()
    print("row ratio between the best and worst mask: %.1fx"
          % (max(rowsurv.values()) / max(min(rowsurv.values()), 1)))

    # Does the same confound sit in G9/G10/G11, the stages that came out flat?
    print()
    print("--- the stages that came out FLAT: rows entering training ---")
    import importlib
    g9 = json.load(open("results-g9/g9_threshold.json"))
    for k, v in sorted(g9["coverage"].items())[:3]:
        frac, ds = k.split(":")
        rs = corpus.load(tp)
        corpus.mask_fraction(rs, float(frac), "cell_drop", seed=int(ds))
        corpus.match_supervised_count(rs, g9["matched_count"], seed=0)
        live = sum(1 for r in rs if r.n_supervised_steps() > 0)
        print("  G9  %-12s rows>=1 = %6d (%.1f%%)" % (k, live, 100*live/len(rs)))

    g11m = importlib.import_module("experiments.g11_breadth")
    g11 = json.load(open("results-g11/g11_breadth.json"))
    for lab, info in g11["coverage"].items():
        rs = corpus.load(tp)
        g11m.throttle(rs, info["buckets"], info["rate"], seed=0)
        corpus.match_supervised_count(rs, g11["matched_count"], seed=0)
        live = sum(1 for r in rs if r.n_supervised_steps() > 0)
        print("  G11 %-12s rows>=1 = %6d (%.1f%%)" % (lab, live, 100*live/len(rs)))


if __name__ == "__main__":
    main()
