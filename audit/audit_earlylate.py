#!/usr/bin/env python
"""Is the early-versus-late contrast matched on everything we can enumerate?

That contrast is the cleanest thing in the dataset: same number of supervised steps,
same surviving rows, same steps per row, 21.6 points apart.  Before leaning on it,
check the things that could still differ and would make it an artefact:

  1. the surviving ROW SETS -- are they the same traces, or different ones?
  2. per-trace supervision -- does one mask concentrate on long traces?
  3. the token count, against the measured quantity bound
  4. for the fkl arms, whether the teacher logit cache covers both masks equally;
     a mask-dependent cache miss would corrupt one arm's training signal
"""
import json, pathlib, collections
import numpy as np
from scd import corpus
from experiments import _common as C

TEACHER = "Qwen/Qwen2.5-7B-Instruct"


def build(mode, floor):
    rs = corpus.load(C.traces_path(TEACHER, 6000, 4))
    corpus.mask_fraction(rs, 0.25, mode, seed=0)
    corpus.match_supervised_count(rs, floor, seed=0)
    return rs


def main():
    g3 = json.load(open("results/g3_measure_vs_loss.json"))
    floor = g3["matched_count"]
    masks = {m: build(m, floor) for m in ("uniform", "early", "late")}

    print("--- 1. surviving row sets ---")
    sets = {}
    for m, rs in masks.items():
        sets[m] = {i for i, r in enumerate(rs) if r.n_supervised_steps() > 0}
        print("  %-9s %6d dong" % (m, len(sets[m])))
    print("  early == late ?  %s" % (sets["early"] == sets["late"]))
    print("  early == uniform ?  %s" % (sets["early"] == sets["uniform"]))

    print()
    print("--- 2. supervision against trace length ---")
    for m, rs in masks.items():
        live = [r for r in rs if r.n_supervised_steps() > 0]
        L = np.array([len(r.steps) for r in live], float)
        S = np.array([r.n_supervised_steps() for r in live], float)
        print("  %-9s mean trace length=%.2f  mean supervised steps=%.2f  "
              "ratio=%.3f  corr(len,sup)=%+.3f"
              % (m, L.mean(), S.mean(), (S / L).mean(),
                 float(np.corrcoef(L, S)[0, 1])))

    print()
    print("--- 3. token counts against the measured bound ---")
    tok = collections.defaultdict(set)
    for r in g3["runs"]:
        tok[r["mode"]].add(r["n_supervised_positions"])
    e, l = min(tok["early"]), min(tok["late"])
    print("  early %d tokens | late %d tokens | ratio %.2fx" % (e, l, e / l))
    print("  measured bound: a fourfold change moves accuracy by <=1.5 points")
    print("  => a 1.49x difference sits at one to two points; the observed gap is 21.6 points")

    print()
    print("--- 4. teacher logit-cache coverage ---")
    d = pathlib.Path("artifacts/logits_Qwen2p5-7B-Instruct")
    if not d.exists():
        print("  (khong tim thay cache)")
        return
    have = {p.stem for p in d.glob("*.npz")} | {p.stem for p in d.glob("*.pt")}
    if not have:
        have = {p.stem for p in d.iterdir() if p.is_file()}
    print("  cache holds %d keys" % len(have))
    from scd import batching
    for m, rs in masks.items():
        live = [r for r in rs if r.n_supervised_steps() > 0]
        keys = [batching.cache_key(r) for r in live]
        hit = sum(1 for k in keys if k in have)
        print("  %-9s %6d/%6d rows with logit cache (%.1f%%)"
              % (m, hit, len(keys), 100 * hit / max(len(keys), 1)))


if __name__ == "__main__":
    main()
