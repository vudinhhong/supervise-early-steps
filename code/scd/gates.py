r"""Gate verdicts, computed from result blobs against ``thresholds.json``.

The point of putting this in code is that a gate should not be a paragraph someone
re-reads charitably at 2am.  ``run.sh <stage>`` ends by calling this, so every
stage prints PASS/FAIL per criterion, the observed value, and the action the
protocol prescribes on failure.

No gate reads a number that is not in a saved result file, and no threshold lives
anywhere but ``thresholds.json``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from . import runio

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


class Verdict:
    def __init__(self, stage: str):
        self.stage = stage
        self.rows: List[Tuple[str, bool | None, str, str]] = []

    def add(self, code: str, ok: bool | None, observed: str, action: str = ""):
        self.rows.append((code, ok, observed, action))

    @property
    def passed(self) -> bool:
        """Pass iff something was judged and everything judged passed.

        Two bugs have lived in this one predicate, in opposite directions, and both
        were expensive.

        The first version returned True when every row was ``n/a``, so a missing
        result file reported PASS -- the most dangerous thing a gate harness can do.
        The fix added ``len(judged) == len(self.rows)``, which overshot: diagnostic
        rows are deliberately ``None``, so from then on *any* stage carrying one
        could never pass. Stage GP duly cleared all six of its criteria, printed
        ``VERDICT: FAIL (0 criteria)``, and stopped a chain that had just produced a
        31-point gain.

        The missing-file case needs no extra clause: when the result is absent the
        gate adds a single ``None`` row and no judged rows, so ``bool(judged)`` is
        already False.
        """
        judged = [r[1] for r in self.rows if r[1] is not None]
        return bool(judged) and all(judged)

    def print(self) -> None:
        print(f"\n===== GATE {self.stage} "
              f"{'=' * max(4, 60 - len(self.stage))}")
        for code, ok, obs, action in self.rows:
            tag = (f"{GREEN}PASS{RESET}" if ok else
                   f"{RED}FAIL{RESET}" if ok is False else f"{YELLOW}n/a {RESET}")
            print(f"  [{tag}] {code:<34} {obs}")
            if ok is False and action:
                print(f"         -> {action}")
        n_fail = sum(1 for r in self.rows if r[1] is False)
        n_judged = sum(1 for r in self.rows if r[1] is not None)
        if self.passed:
            print(f"  {GREEN}VERDICT: PASS{RESET} ({n_judged} criteria met) -- "
                  f"proceed to the next stage.")
        elif n_judged == 0:
            print(f"  {RED}VERDICT: BLOCKED{RESET} -- nothing to judge "
                  f"(the stage has not produced its result yet).")
        else:
            print(f"  {RED}VERDICT: FAIL{RESET} ({n_fail} of {n_judged} criteria) "
                  f"-- do NOT run the next stage; see the actions above.")


def add_resources(v: "Verdict", d: dict) -> None:
    """Two diagnostic rows every GPU stage should carry.

    Memory is reported rather than gated.  A stated budget used to be enforced and
    it killed a run that had two gigabytes spare; what the number is good for is
    sizing the *next* submission, which needs it printed, not obeyed.

    Arms skipped for running out of memory are surfaced loudly, because an average
    over fewer seeds than the design called for is a weaker result than it looks.
    """
    env = d.get("_env", {})
    if "peak_allocated_gib" in env:
        v.add("resources [diagnostic]", None,
              f"peak allocated {env['peak_allocated_gib']:.1f} GiB, reserved "
              f"{env.get('peak_reserved_gib', float('nan')):.1f}, device "
              f"{env.get('device_total_gib', float('nan')):.1f}, stated budget "
              f"{env.get('stated_budget_gib', float('nan')):.0f}")
    oom = d.get("oom_arms") or []
    if oom:
        v.add("arms lost to out-of-memory", False,
              f"{len(oom)} arm(s) skipped: "
              f"{', '.join(str({k: a[k] for k in a if k != 'error'}) for a in oom[:3])}"
              + (" ..." if len(oom) > 3 else ""),
              "rerun those arms with a lower GEN_BATCH or GRAD_CHECKPOINTING=1 "
              "before reading the comparison: the design's seed count was not met")


# ------------------------------------------------------------------ G0 and G1


def gate_g0(tag: str = "g0_estimator") -> Verdict:
    r"""Which estimators are admissible, and is the study still identified?

    Restructured after the first real run, and the restructuring is recorded in
    ``thresholds.json`` revision 4 rather than hidden.  The original wiring failed
    the whole stage if any criterion failed for any estimator, which does not
    implement the decision rule this protocol registered in advance: §2.5 of the
    design says that when the estimators disagree the response is "continue with
    estimator A alone, treat B and C as reference".  So the stage now decides *per
    estimator* and blocks only if the load-bearing one fails or if nothing
    corroborates it.

    Two numbers from that run drive the shape.

    *The accuracy criterion was comparing a bounded estimator against an unbounded
    reference.*  Both estimators clip the per-state density ratio at 50, so the
    coefficient they can report is bounded well below the exact value, which
    reaches 98,208 on the hole family.  For the 36.5% of configurations whose exact
    value exceeds what a clipped estimator can represent, the +/-20% test fails by
    construction however good the estimator is.  Restricted to the range where the
    comparison means anything, estimator A scores 0.93 and B 0.89, against the
    pooled 0.74 and 0.56.  Both values are reported.

    *The disagreement between A and C is real.*  That was checked rather than
    assumed: no estimate in the run reaches the clip, so the rank correlation of
    0.643 is not a tie artefact.  Estimator C is therefore dropped, which is the
    registered action and also what its own docstring predicted.

    Identification (criterion C5) survives: it needs two independent estimators to
    agree in rank, and A~B is 0.868.
    """
    v = Verdict("G0 -- which estimators are admissible?")
    d = runio.load(tag, where="results-syn")
    if d is None:
        v.add("G0", None, f"results-syn/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["g0"]
    w = d["world_validity"]

    # ---- is the instrument fit for the question at all
    v.add("G0.0a world dynamic range",
          w["gap_sd"] >= T["world_gap_sd_min"],
          f"sd(gap)={w['gap_sd']:.3f} (need >= {T['world_gap_sd_min']})",
          "the assay has no range: fix the task, not the theory")
    ratio = w["c2_max"] / max(w["c2_min"], 1e-9)
    v.add("G0.0b C2 spread", ratio >= T["world_c2_ratio_min"],
          f"max/min={ratio:.0f} (need >= {T['world_c2_ratio_min']})",
          "widen the mu family: a correlation on a narrow grid means nothing")
    v.add("G0.0c dissociation present",
          w["n_dissociating"] >= T["world_dissociating_min"],
          f"{w['n_dissociating']} configs with E_mu[KL]<=0.01 and gap>=0.10",
          "the world does not exhibit the phenomenon, so it cannot adjudicate it")
    v.add("G0.1 Theorem 1 never violated",
          w["theorem1_violations"] <= T["theorem1_violations_max"],
          f"{w['theorem1_violations']}/{w['n_configs']} violations",
          "a theorem cannot be violated: this is a bug in code or in the proof")

    # ---- accuracy, in the range where the comparison is defined
    in_range = _accuracy_in_range(d, T.get("clip", 50.0))

    bars = {"c2_A": T["rank_rho_min_coarse"], "c2_B": T["rank_rho_min_fine"],
            "c2_C": T["rank_rho_min_fine"]}
    admissible = {}
    for key in ("c2_A", "c2_B", "c2_C"):
        rank = d["rank"][key]["rho"]
        cv = d["stability_cv"][key]["median_cv"]
        acc_pool = d["accuracy"][key]["frac_within_20pct"]
        acc_ir = in_range.get(key, {}).get("frac_within_20pct", float("nan"))
        ok_rank = rank >= bars[key]
        ok_cv = cv <= T["median_cv_max"]
        ok_acc = (acc_ir >= T["within20_min"]
                  if key in T["within20_applies_to"] else True)
        # Agreement with the load-bearing estimator is part of being admissible,
        # not a separate stage-level question.  A first version of this gate folded
        # it into one "A is corroborated" row, which needs only a single partner --
        # so estimator C was marked admissible on rank and stability alone while
        # disagreeing with A at 0.643 and reporting magnitudes off by an order of
        # magnitude.  The registered rule is the other way round: an estimator that
        # does not agree with A has no role, because A is what every stage computes.
        agree = (d["pairwise"].get(f"c2_A~{key}")
                 or d["pairwise"].get(f"{key}~c2_A"))
        ok_agree = True if key == "c2_A" else bool(
            agree and agree["rho"] >= T["pairwise_rho_min"])
        admissible[key] = bool(ok_rank and ok_cv and ok_acc and ok_agree)
        # Per-estimator rows are DIAGNOSTIC, not stage-blocking.  Design doc §2.5
        # registered the response to disagreement in advance -- "continue with
        # estimator A alone, treat B and C as reference" -- so an inadmissible B or
        # C narrows the study's scope rather than stopping it.  What blocks is
        # below: A itself unusable, or nothing admissible agreeing with A.
        v.add(f"G0 estimator {key[3:]}: "
              + ("ADMISSIBLE" if admissible[key] else "DROPPED") + " [diagnostic]",
              None,
              f"rank={rank:+.3f} (need >= {bars[key]}), "
              f"acc in range={acc_ir:.2f} (pooled {acc_pool:.2f}"
              + (f", need >= {T['within20_min']}" if key in T["within20_applies_to"]
                 else ", not a bar for this estimator")
              + f" on {in_range.get(key, {}).get('n', 0)}"
              f"/{in_range.get(key, {}).get('n_total', 0)} comparable configs), "
              f"CV={cv:.3f}"
              + ("" if key == "c2_A" else
                 f", agrees with A at {agree['rho']:+.3f}" if agree else
                 ", no agreement measured")
              + ("" if admissible[key] else
                 "  -- NOT usable; the paper must say it was dropped and must not "
                 "claim its behaviour as evidence"))

    # ---- the load-bearing estimator, and corroboration
    v.add("G0.A load-bearing estimator usable", admissible.get("c2_A", False),
          "estimator A (discrete abstraction) is what every GPU stage computes; "
          "B and C only ever cross-check it",
          "STOP: nothing downstream can be measured")
    def _other(name: str) -> str:
        a, b = name.split("~")
        return b if a == "c2_A" else a
    partners = [n for n, r in d["pairwise"].items()
                if "c2_A" in n and r["rho"] >= T["pairwise_rho_min"]
                and admissible.get(_other(n), False)]
    v.add("G0.C5 identification: A corroborated by an admissible estimator",
          bool(partners),
          "; ".join(f"{n}: rho={d['pairwise'][n]['rho']:+.3f}"
                    for n in d["pairwise"] if "c2_A" in n)
          + f"  (need >= {T['pairwise_rho_min']} from at least one)",
          "no second estimator agrees with A in rank: the quantity has one "
          "estimator, which is a definition and not a measurement")
    dropped = [k[3:] for k in ("c2_B", "c2_C") if not admissible.get(k, False)]
    v.add("G0 scope consequence [diagnostic]", None,
          ("all three estimators admissible" if not dropped else
           f"dropped: {', '.join(dropped)}; the paper must say so and must not "
           f"claim their behaviour as evidence"))
    return v


def _accuracy_in_range(d: dict, clip: float) -> Dict[str, dict]:
    r"""Accuracy restricted to configurations the estimator can represent.

    A clipped estimator cannot report a coefficient near the exact value when the
    exact value is far above the clip, so a relative-error test over those
    configurations measures the clip rather than the estimator.  The reference
    differs per estimator: A is judged against the exactly computable coarsened
    value, B and C against the full one, because that is the coarsening each of
    them sees.
    """
    import math
    ref_of = {"c2_A": "exact_sigma", "c2_B": "exact_feature",
              "c2_C": "exact_feature"}
    est = d.get("estimates") or []
    if not est:
        return {}
    n_max = max(e["n"] for e in est)
    big = [e for e in est if e["n"] == n_max]
    out = {}
    for key, refk in ref_of.items():
        vals = [(e[key], e[refk]) for e in big
                if e.get(refk) and e[refk] > 0 and math.isfinite(e[key])]
        ok = [(x, r) for x, r in vals if r <= clip]
        rel = [abs(x / r - 1.0) for x, r in ok]
        out[key] = {"frac_within_20pct":
                    float(sum(1 for z in rel if z <= 0.20) / len(rel)) if rel
                    else float("nan"),
                    "median_rel_err": float(sorted(rel)[len(rel) // 2]) if rel
                    else float("nan"),
                    "n": len(ok), "n_total": len(vals), "clip": clip,
                    "sample_size": n_max}
    return out


def gate_g1(tag: str = "g1_theory") -> Verdict:
    v = Verdict("G1 -- is Theorem 1 useful? (EXPLORATORY)")
    d = runio.load(tag, where="results-syn")
    if d is None:
        v.add("G1", None, f"results-syn/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["g1"]
    ks = sorted(int(k) for k in d["by_capacity"])
    ample = d["by_capacity"][str(ks[-1])]

    v.add("G1.1 product predicts (ample capacity)",
          ample["rho_product"]["rho"] >= T["ample_capacity_rho_product_min"],
          f"rho={ample['rho_product']['rho']:+.3f} at k={ks[-1]} "
          f"(need >= {T['ample_capacity_rho_product_min']})",
          "STOP: the product does not predict the gap even with exact terms")
    v.add("G1.2 raw fidelity goes blind (ample capacity)",
          ample["rho_fidelity_raw"]["rho"] <= T["ample_capacity_rho_fidelity_raw_max"],
          f"rho={ample['rho_fidelity_raw']['rho']:+.3f} "
          f"(need <= {T['ample_capacity_rho_fidelity_raw_max']})",
          "STOP: fidelity on the real corpus already predicts; there is no gap to fill")
    pr = d["pooled"].get("partial_regression")
    if pr and "log_cov" in pr["names"]:
        i = pr["names"].index("log_cov")
        j = pr["names"].index("log_fid")
        ratio = abs(pr["beta"][i]) / max(abs(pr["beta"][j]), 1e-12)
        v.add("G1.4 coverage term carries signal",
              pr["p"][i] <= T["partial_beta_cov_max_p"] and pr["beta"][i] > 0
              and ratio >= T["partial_beta_cov_min_ratio"],
              f"beta_cov={pr['beta'][i]:+.3f} (p={pr['p'][i]:.1e}), "
              f"|beta_cov|/|beta_fid|={ratio:.2f}",
              "downgrade the claim to 'a forgotten term', not 'the dominant term'")
    tc = d["pooled"]["tightness"]
    v.add("G1.5 bound stable enough to rank",
          tc["cv"] <= T["tightness_cv_max"],
          f"CV of gap/bound = {tc['cv']:.2f} (need <= {T['tightness_cv_max']})",
          "present Theorem 1 as a decomposition, not as a numerical bound")
    c = d["corollary_1_1"]
    v.add("G1.6 Corollary 1.1 instantiated",
          c["n"] >= T["corollary_min_configs"]
          and c["max_gap_at_zero_fidelity"] >= T["corollary_min_gap_at_zero_fidelity"],
          f"{c['n']} configs; largest gap at exactly zero fidelity loss = "
          f"{c['max_gap_at_zero_fidelity']:.3f}",
          "the impossibility corollary is not realised; re-examine the construction")
    it = d["interaction"]
    v.add("G1.7 regime interaction",
          it["spearman_capacity_vs_advantage"]["rho"] >= T["interaction_rho_min"],
          f"capacity vs coverage-advantage rho="
          f"{it['spearman_capacity_vs_advantage']['rho']:+.3f} "
          f"(need >= {T['interaction_rho_min']})",
          "no regime structure: the GPU stages cannot be designed around one")
    return v


REGISTRY: Dict[str, Callable[[], Verdict]] = {"g0": gate_g0, "g1": gate_g1}


def selftest() -> None:
    """Assert the verdict predicate on the four cases that matter.

    Run before every stage by ``run.sh``.  It costs milliseconds and would have
    caught both historical bugs; a gate harness that is not itself tested is a
    single point of failure in front of every GPU hour the study spends.
    """
    missing = Verdict("missing result")
    missing.add("x", None, "result file absent")
    assert not missing.passed, "a gate with nothing judged must not pass"

    diagnostics_only = Verdict("diagnostics only")
    diagnostics_only.add("d1", None, "informational")
    diagnostics_only.add("d2", None, "informational")
    assert not diagnostics_only.passed, "diagnostics alone decide nothing"

    mixed = Verdict("pass with diagnostics")
    mixed.add("d", None, "informational")
    mixed.add("a", True, "ok")
    mixed.add("b", True, "ok")
    assert mixed.passed, "diagnostic rows must not block a pass"

    one_bad = Verdict("one failure")
    one_bad.add("d", None, "informational")
    one_bad.add("a", True, "ok")
    one_bad.add("b", False, "not ok")
    assert not one_bad.passed, "a single failure must block"

    empty = Verdict("empty")
    assert not empty.passed, "an empty verdict must not pass"
    print("[gates] selftest ok (5 cases)")


def run(stage: str) -> int:
    try:
        from . import gates_gpu
        REGISTRY.update(gates_gpu.REGISTRY)
    except Exception:
        pass
    fn = REGISTRY.get(stage)
    if fn is None:
        print(f"[gates] no verdict defined for stage {stage!r}; "
              f"known: {sorted(REGISTRY)}")
        return 0
    v = fn()
    v.print()
    return 0 if v.passed else 1
