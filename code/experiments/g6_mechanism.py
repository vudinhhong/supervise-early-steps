r"""G6 -- does the selection rule do what it is supposed to, before anything is trained?

G6.1 is the cheapest important measurement in the study and it runs *before* G3 and
G5.  It asks: does branching from prefixes chosen by the density ratio lower the
coverage coefficient more than branching from random prefixes, at the same number
of teacher calls?  If not, the intervention does not do the thing the theory says it
does, and no amount of training will rescue it -- which is precisely the position
the compute-matched study of the previous round found itself in after building three
mechanisms.

The measurement must be made on prefixes *held out* from the selection, or it is
circular: choosing prefixes with a high ratio and then measuring the ratio on those
same prefixes shows a reduction by construction (validity condition V7).

G6.4 runs three alternative selection rules if the primary one fails, so a failure
of one heuristic is not mistaken for a failure of the idea.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from scd import corpus, coverage, hf, runio, sampling, stats
from scd.abstraction import N_STATES
from experiments import _common as C
from experiments.g5_main import branch_items


def alt_select(rows, teacher_rows, budget, how, seed=0, extra=None):
    """Alternatives for G6.4: rarest cell, k-center on hidden states, entropy."""
    cand = [(i, t) for i, r in enumerate(rows) for t in range(1, len(r.steps))]
    codes = np.array([rows[i].sigma()[t] for i, t in cand])
    rng = np.random.default_rng(seed)
    if how == "rarest_cell":
        mu = np.bincount(np.asarray(corpus.measure_codes(rows)), minlength=N_STATES)
        score = -mu[codes].astype(float)
    elif how == "deficit_cell":
        d = np.bincount(np.asarray(corpus.teacher_codes(teacher_rows)), minlength=N_STATES) + 0.5
        mu = np.bincount(np.asarray(corpus.measure_codes(rows)), minlength=N_STATES) + 0.5
        score = (d / d.sum() - mu / mu.sum())[codes]
    elif how == "entropy":
        # Alternative (c): where the student is least certain.  Scores are
        # precomputed in main() so the student is loaded once, not per budget.
        ent = (extra or {}).get("entropy")
        if ent is None:
            raise ValueError("entropy rule needs precomputed student entropies")
        score = np.array([ent[i][t] if t < len(ent[i]) else -np.inf
                          for i, t in cand], dtype=float)
    elif how == "uniform_cell":
        # one branch per abstraction cell, round-robin: maximal cell coverage
        order, seen = [], {}
        for j, c in enumerate(codes):
            seen.setdefault(int(c), []).append(j)
        cells = list(seen)
        rng.shuffle(cells)
        while len(order) < budget and any(seen[c] for c in cells):
            for c in cells:
                if seen[c] and len(order) < budget:
                    order.append(seen[c].pop())
        sel = np.array(order[:budget])
        return [{"question": rows[cand[int(j)][0]].question,
                 "gold": rows[cand[int(j)][0]].gold,
                 "source": rows[cand[int(j)][0]].source,
                 "prefix_steps": rows[cand[int(j)][0]].steps[:cand[int(j)][1]],
                 "parent": cand[int(j)][0]} for j in sel]
    else:
        raise ValueError(how)
    sel = np.argsort(-(score + rng.uniform(0, 1e-9, size=len(score))))[:budget]
    return [{"question": rows[cand[int(j)][0]].question,
             "gold": rows[cand[int(j)][0]].gold,
             "source": rows[cand[int(j)][0]].source,
             "prefix_steps": rows[cand[int(j)][0]].steps[:cand[int(j)][1]],
             "parent": cand[int(j)][0]} for j in sel]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompts", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--budgets", default="100 200 400 800")
    ap.add_argument("--max-budget-frac", type=float, default=0.5)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--rules",
                    default="ratio random rarest_cell deficit_cell uniform_cell "
                            "entropy")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--mode", default="cell_drop",
                    help="how the deficient measure is built. cluster_drop bans "
                         "cells rarest-first by count mass and at 0.25 leaves ONE "
                         "cell, which makes the comparison meaningless")
    ap.add_argument("--keep-cells", type=float, default=0.5)
    ap.add_argument("--min-cell-coverage", type=float, default=0.20)
    ap.add_argument("--max-cell-coverage", type=float, default=0.80)
    ap.add_argument("--tag", default="g6_mechanism")
    a = ap.parse_args()

    rows = C.build_traces(a.teacher, a.prompts, a.samples, gen_batch=a.gen_batch)

    # V7: split rows into a selection half and a held-out half.  The coefficient is
    # measured on the held-out half only.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    n_hold = int(a.holdout * len(rows))
    hold = [rows[int(i)] for i in idx[:n_hold]]
    sel_pool = [rows[int(i)] for i in idx[n_hold:]]
    corpus.mask_fraction(sel_pool, a.keep_cells, a.mode, seed=0)
    corpus.mask_fraction(hold, a.keep_cells, a.mode, seed=0)

    before_sel = C.coverage_of(sel_pool, rows)
    before_hold = C.coverage_of(hold, rows)

    # Precondition, and the one this stage was missing.
    #
    # The question is where to spend a budget, so the supervision has to be
    # PARTIALLY deficient: a measure covering one cell of 413 is annihilated, not
    # deficient, and from there any way of adding four thousand diverse prefixes
    # covers three to four hundred cells -- random scored 1.797 against the ratio
    # rule's 1.912 on exactly that measure, which is the same answer twice.  Three
    # and a half GPU-hours went into a comparison that could not come out.
    #
    # The log had said so all along: n_states_seen_mu was 1.  Nobody read it, so
    # it is a gate now.
    n_cells_d = before_hold["n_states_seen_d"]
    frac_cells = before_hold["n_states_seen_mu"] / max(n_cells_d, 1)
    print(f"[g6] measure covers {before_hold['n_states_seen_mu']} of "
          f"{before_hold['n_states_seen_d']} cells the teacher visits "
          f"({frac_cells:.1%})")
    if not (a.min_cell_coverage <= frac_cells <= a.max_cell_coverage):
        runio.save(a.tag, {"args": vars(a), "cell_coverage": frac_cells, "before": {"selection": before_sel,
                           "heldout": before_hold},
                           "cell_coverage": frac_cells, "rules": {},
                           "aborted": f"measure covers {frac_cells:.1%} of the "
                           f"teacher's cells; need between "
                           f"{a.min_cell_coverage:.0%} and "
                           f"{a.max_cell_coverage:.0%}"})
        print("[g6] STOPPING before the branch sweep: the measure is not "
              "partially deficient, so which prefix a rule picks cannot matter.\n"
              "     Adjust --keep-cells / --mode and rerun.")
        return
    print(f"[g6] before: C2(selection half)={before_sel['c2']:.3f}  "
          f"C2(held out)={before_hold['c2']:.3f}")

    # Sweep the budget instead of fixing it, and the reason is arithmetic.
    #
    # The first valid run used 4000 branches against 243 missing cells out of 413.
    # Covering n cells by random draws takes about n*ln(n) = 2488 draws, so 4000 is
    # 1.6x the coupon-collector threshold: random selection covers essentially
    # everything and no rule can distinguish itself.  The measured cell counts said
    # so -- 334 to 408 of 413 for every rule -- and every rule landed on a held-out
    # coefficient between 1.23 and 1.60.
    #
    # Selection can only matter where the budget is SCARCE relative to the gap.  A
    # sweep is also cheaper than the single large point it replaces: 100+200+400+800
    # is 1500 branches per rule against 4000.
    coupon = n_cells_d * math.log(max(n_cells_d, 2))
    budgets = sorted(int(x) for x in a.budgets.split())
    print(f"[g6] coupon-collector threshold for {n_cells_d} cells = {coupon:.0f} "
          f"draws; budgets {budgets}")
    if max(budgets) > a.max_budget_frac * coupon:
        runio.save(a.tag, {"args": vars(a), "cell_coverage": frac_cells,
                           "coupon_threshold": coupon, "rules": {}, "sweep": {},
                           "aborted": f"largest budget {max(budgets)} exceeds "
                           f"{a.max_budget_frac:.0%} of the coupon-collector "
                           f"threshold {coupon:.0f}; random selection saturates and "
                           f"the comparison cannot discriminate"})
        print("[g6] STOPPING: the budget saturates random selection.")
        return

    # Student entropies, once, before the teacher is resident.
    extra = {}
    if "entropy" in a.rules.split():
        st_tok = hf.load_tokenizer(a.student)
        st_model = hf.load_model(a.student)
        print(f"[g6] computing student entropies over {len(sel_pool)} traces")
        extra["entropy"] = sampling.entropy_at_prefixes(
            st_model, st_tok, sel_pool, micro_batch=8)
        hf.free(st_model)

    tok = hf.load_tokenizer(a.teacher)
    model = hf.load_model(a.teacher)
    sweep, out = {}, {}
    for budget in budgets:
        per_rule = {}
        for rule in a.rules.split():
            items = (branch_items(sel_pool, rows, budget, rule, seed=1)
                     if rule in ("ratio", "random")
                     else alt_select(sel_pool, rows, budget, rule, seed=1,
                                     extra=extra))
            new = sampling.continue_from_prefix(model, tok, items,
                                               micro_batch=a.gen_batch, seed=1)
            add = [r for r in corpus.from_records(new, origin="branch") if r.correct]
            corpus.mask_all(add)
            after_hold = C.coverage_of(hold + add, rows)
            per_rule[rule] = {
                "requested": len(items), "kept": len(add),
                "c2_heldout_after": after_hold["c2"],
                "reduction_heldout": before_hold["c2"] - after_hold["c2"],
                "cells_covered_after": after_hold["n_states_seen_mu"]}
            print(f"[g6] budget {budget:5d}  {rule:14s} held-out C2 "
                  f"{before_hold['c2']:.3f} -> {after_hold['c2']:.3f}  "
                  f"cells {after_hold['n_states_seen_mu']:3d}")
        rn = per_rule.get("random", {}).get("reduction_heldout", float("nan"))
        sweep[str(budget)] = {
            "per_rule": per_rule,
            "ratio_over_random": (per_rule.get("ratio", {})
                                  .get("reduction_heldout", float("nan")) / rn)
            if rn else float("inf"),
            "best_rule": max(per_rule, key=lambda k:
                             per_rule[k]["reduction_heldout"]) if per_rule else None,
            "best_over_random": (max(v["reduction_heldout"]
                                     for v in per_rule.values()) / rn)
            if rn else float("inf")}
        out = per_rule          # the largest budget, for backward compatibility
    hf.free(model)

    best_b = max(sweep, key=lambda b: sweep[b]["ratio_over_random"]) if sweep else None
    runio.save(a.tag, {"args": vars(a), "cell_coverage": frac_cells,
                       "coupon_threshold": coupon,
                       "before": {"selection": before_sel, "heldout": before_hold},
                       "sweep": sweep, "rules": out,
                       "ratio_over_random":
                           sweep[best_b]["ratio_over_random"] if best_b else float("nan"),
                       "best_budget": int(best_b) if best_b else None,
                       "best_rule": sweep[best_b]["best_rule"] if best_b else None,
                       "best_over_random":
                           max((v["best_over_random"] for v in sweep.values()),
                               default=float("nan"))})
    print("\n[g6] ratio/random by budget: " + ", ".join(
        f"{b}: {v['ratio_over_random']:.3f}" for b, v in sweep.items()))
    print("[g6] best rule by budget:    " + ", ".join(
        f"{b}: {v['best_rule']} ({v['best_over_random']:.3f}x)"
        for b, v in sweep.items()))
    print("[g6] criterion G6.1 needs ratio/random >= 1.5 at some budget")


if __name__ == "__main__":
    main()
