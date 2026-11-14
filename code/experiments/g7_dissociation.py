r"""G7 -- prompt coverage or state coverage?

The field has a result that looks fatal to any coverage account: a single training
query recovers most of the full-data outcome across four domains.  If coverage of
the *task* were the mechanism, that could not happen.

This separates two things that the literature conflates.  The number of distinct
prompts and the number of distinct reasoning states are different quantities, and
one query sampled many times produces many distinct states.  Holding the number of
distinct abstraction cells fixed while varying the number of prompts over four
orders of magnitude says which one the student is actually consuming.

Validity condition V8: with one prompt, the arm must genuinely reach many distinct
cells.  If 200k supervised positions collapse onto thirty cells, that arm has no
coverage and the comparison is empty -- so the census of distinct cells is reported
for every arm and the gate reads it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np

from scd import corpus, data, hf, runio, sampling, stats, train
from scd.abstraction import N_STATES
from experiments import _common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--student", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompt-counts", default="1 10 100 1000 6000")
    ap.add_argument("--position-budgets", default="25000 100000")
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--suite-limit", type=int, default=0)
    ap.add_argument("--tag", default="g7_dissociation")
    ap.add_argument("--runs-tag", default="",
                    help="whose <tag>_runs.jsonl to append to and re-read before "
                         "every arm, so a restart resumes instead of repeating")
    C.shard_args(ap)
    a = ap.parse_args()

    counts = [int(x) for x in a.prompt_counts.split()]
    budgets = [int(x) for x in a.position_budgets.split()]
    seeds = [int(s) for s in a.seeds.split()]
    problems, census = C.suite(a.suite_limit)

    # The ten corpora are cached on disk.  Sampling them costs more than several
    # training arms, and without a cache every restart of this stage would pay for
    # them again.  The teacher is only loaded if something is actually missing.
    art = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
    cpath = {(np_, B): art / f"g7_corpus_p{np_}_b{B}.jsonl"
             for np_ in counts for B in budgets}
    missing = [k for k, v in cpath.items() if not v.exists()]

    teacher = tok_t = pool = None
    if missing:
        print(f"[g7] {len(missing)} of {len(cpath)} corpora missing; "
              f"loading the teacher to sample them")
        tok_t = hf.load_tokenizer(a.teacher)
        teacher = hf.load_model(a.teacher)
        pool = data.train_questions(limit=max(counts))
    else:
        print(f"[g7] all {len(cpath)} corpora cached; the teacher is not needed")

    corpora = {}
    for np_ in counts:
        for B in budgets:
            path = cpath[(np_, B)]
            if path.exists():
                rows = corpus.load(path)
                corpus.mask_all(rows)
                got = corpus.match_supervised_count(rows, B, seed=0)
                cells = len(set(corpus.measure_codes(rows)))
                corpora[(np_, B)] = rows
                print(f"[g7] prompts={np_:5d} budget={B:6d}: {len(rows)} traces, "
                      f"{got} positions, {cells}/{N_STATES} distinct cells "
                      f"(cached)")
                continue
            # samples per prompt chosen so the total supervised positions matches B
            per = min(a.max_samples, max(1, round(B / (np_ * 6))))
            recs = sampling.sample_traces(teacher, tok_t, pool[:np_], per,
                                          micro_batch=a.gen_batch, seed=0)
            rows = [r for r in corpus.from_records(recs) if r.correct]
            corpus.save(rows, path)
            corpus.mask_all(rows)
            got = corpus.match_supervised_count(rows, B, seed=0)
            cells = len(set(corpus.measure_codes(rows)))
            corpora[(np_, B)] = rows
            print(f"[g7] prompts={np_:5d} budget={B:6d}: {len(rows)} traces, "
                  f"{got} positions, {cells}/{N_STATES} distinct cells, "
                  f"{per} samples/prompt")
    if teacher is not None:
        hf.free(teacher)

    # The teacher's own trace pool, which is what C2 has to be measured against.
    base_rows = corpus.load(C.traces_path(a.teacher, 6000, 4))

    # Coverage per corpus, measured once and before any training.  It used to be
    # computed inside the corpus loop; with one arm per job the loop is flat, so it
    # moves out here rather than being redone for every seed.
    covs, cellcounts = {}, {}
    for key, rows in corpora.items():
        # Against the teacher's FULL occupancy, not against itself.  The earlier
        # self-comparison made C2 identically 1.00 for every arm, which hid the
        # fact that this sweep moves C2 over a range of 8x: one prompt sampled
        # many times reaches few cells AND sits far from the teacher's occupancy.
        covs[key] = C.coverage_of(rows, base_rows)
        cellcounts[key] = len(set(corpus.measure_codes(rows)))

    arms = C.shard([(k, int(sd)) for k in corpora for sd in seeds],
                   a.nshards, a.shard)
    if a.list_arms:
        print("[g7] shard %d/%d: %d arm(s)" % (a.shard, a.nshards, len(arms)))
        for i, (k, sd) in enumerate(arms):
            print("  %2d  prompts=%-5d budget=%-7d seed=%d  C2=%9.6f"
                  % (i, k[0], k[1], sd, covs[k]["c2"]))
        return

    if a.base_accuracy is not None:
        base = {"accuracy": float(a.base_accuracy)}
        print("[g7] base %.2f%% (given, not measured)" % (100 * base["accuracy"]))
    else:
        m, tok = C.student(a.student, a.lora_r)
        base = C.score(m, tok, problems, a.eval_batch)
        hf.free(m)
        print(f"[g7] base {base['accuracy']*100:.2f}%")

    runs_tag = a.runs_tag or a.tag
    ledger = runio.results_dir() / f"{runs_tag}_runs.jsonl"

    def done_arms() -> set:
        """Arms already recorded.  Re-read before every arm so a second worker on
        another device, or a restart, sees what has landed since startup."""
        out = set()
        if not ledger.exists():
            return out
        for line in ledger.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out.add((int(r["prompts"]), int(r["budget"]), int(r["seed"])))
            except Exception:
                continue
        return out

    runs, oom = [], []
    for line in (ledger.read_text().splitlines() if ledger.exists() else []):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (int(r["prompts"]), int(r["budget"]), int(r["seed"])) in {
                (np_, B, s) for np_ in counts for B in budgets for s in seeds}:
            runs.append(r)
    if runs:
        # Loaded back so the regression below is fitted on the whole design and not
        # on whatever this process happened to train.
        seen_k, ded = set(), []
        for r in runs:
            k = (int(r["prompts"]), int(r["budget"]), int(r["seed"]))
            if k not in seen_k:
                seen_k.add(k); ded.append(r)
        runs = ded
        print(f"[g7] resumed {len(runs)} arms from {ledger.name}")

    def checkpoint(partial: bool = True) -> None:
        runio.save(a.tag, {"args": vars(a), "census": census,
                           "suite_n": len(problems),
                           "base_accuracy": base["accuracy"], "runs": runs,
                           "oom_arms": oom, "partial": partial,
                           "arms_done": len(runs)})

    for _key, seed in arms:
        np_, B = _key
        rows = corpora[_key]
        cov = covs[_key]
        cells = cellcounts[_key]
        if (int(np_), int(B), int(seed)) in done_arms():
            print(f"[g7] skip prompts={np_} budget={B} s{seed} (already done)")
            continue

        def arm(seed=seed, rows=rows):
            tr, probe, fit = C.probe_and_fit_rows(rows)
            # Seed the adapter too, not only the data order; see C.student.
            model, tok = C.student(a.student, a.lora_r, seed=seed)
            cfg = train.TrainConfig(objective="ce", lr=a.lr, epochs=a.epochs,
                                    batch=a.batch, accum=a.accum,
                                    lora_r=a.lora_r, seed=seed)
            info = train.train(model, tok, tr, cfg, probe_rows=probe,
                               fit_rows=fit)
            sc = C.score(model, tok, problems, a.eval_batch)
            hf.free(model)
            return info, sc

        got, err = C.oom_safe(f"g7 p{np_} b{B} s{seed}", arm)
        if got is None:
            oom.append({"prompts": np_, "budget": B, "seed": seed,
                        "error": err})
            continue
        info, sc = got
        runs.append({"prompts": np_, "budget": B, "seed": seed,
                     "distinct_cells": cells, "c2": cov["c2"],
                     "accuracy": sc["accuracy"],
                     "gain_points": 100 * (sc["accuracy"] - base["accuracy"]),
                     "n_supervised_positions": info["n_supervised_positions"],
                     "own_supervision_loss_drop": info["own_supervision_loss_drop"]})
        C.append_run(runs_tag, runs[-1])
        checkpoint()
        print(f"[g7] prompts={np_:5d} budget={B:6d} s{seed}: "
              f"gain {runs[-1]['gain_points']:+.2f} pts")

    y = np.array([r["gain_points"] for r in runs])
    reg = stats.partial_regression(
        y, {"log_cells": np.log([max(r["distinct_cells"], 1) for r in runs]),
            "log_prompts": np.log([max(r["prompts"], 1) for r in runs])})
    i_cells = reg["names"].index("log_cells")
    i_prom = reg["names"].index("log_prompts")
    ratio = abs(reg["beta"][i_cells]) / max(abs(reg["beta"][i_prom]), 1e-12)

    one = [r["gain_points"] for r in runs if r["prompts"] == min(counts)
           and r["budget"] == max(budgets)]
    many = [r["gain_points"] for r in runs if r["prompts"] == max(counts)
            and r["budget"] == max(budgets)]

    runio.save(a.tag, {"args": vars(a), "census": census,
                       "suite_n": len(problems), "partial": False,
                       "oom_arms": oom,
                       "base_accuracy": base["accuracy"], "runs": runs,
                       "regression": reg, "beta_ratio_cells_over_prompts": ratio,
                       "min_distinct_cells": int(min(r["distinct_cells"] for r in runs)),
                       "one_prompt_penalty_points":
                           float(np.mean(many) - np.mean(one)) if one and many
                           else float("nan")})
    print(f"\n[g7] |beta_cells| / |beta_prompts| = {ratio:.2f} "
          f"(criterion needs >= 3.0)")


if __name__ == "__main__":
    main()
