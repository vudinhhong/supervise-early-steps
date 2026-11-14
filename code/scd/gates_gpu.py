r"""Gate verdicts for the stages that need a GPU.

Same discipline as ``scd.gates``: every number comes from a saved result blob and
every threshold from ``thresholds.json``.
"""

from __future__ import annotations

import numpy as np

from . import data, runio
from .gates import Verdict, add_resources


def gate_gp(tag: str = "gp_control") -> Verdict:
    v = Verdict("GP -- positive control (validates the dependent variable)")
    d = runio.load(tag, where="results")
    if d is None:
        v.add("GP", None, f"results/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["gp"]
    # GP.1 is a diagnostic, not a gate (see thresholds.json, revision 2): an
    # absolute point threshold is not well-posed when the ceiling varies by
    # configuration.  The gate is GP.6, which is derived from the reliability
    # calculation rather than calibrated by analogy.
    v.add("GP.1 capability gain [diagnostic]", None,
          f"{d['gain_points']:+.2f} pts "
          f"(reference {T['gain_min_points_diagnostic']}; not a gate)")
    rec = d.get("private_value_recovery", float("nan"))
    if np.isfinite(rec):
        v.add("GP.8 private value recovery [diagnostic]", None,
              f"{100*rec:.1f}% of Gamma={d.get('gamma_points', float('nan')):.1f} pts "
              f"(reference {100*T['recovery_min_diagnostic']:.0f}%; not a gate)")
    else:
        v.add("GP.8 private value recovery [diagnostic]", None,
              "teacher not evaluated: rerun with --measure-teacher 1")
    fh = d.get("format_learning_headroom_points")
    if fh is not None:
        v.add("GP.9 format-learning headroom [diagnostic]", None,
              f"best base prompting beats zero-shot chat by {fh:+.2f} pts "
              f"(reference used: {d.get('base_reference_prompting', '?')}). "
              f"A large value means a zero-shot baseline would have overstated "
              f"the gain by that much")
    v.add("GP.2 all seeds positive", bool(d["all_seeds_positive"]),
          f"{sum(1 for r in d['runs'] if r['accuracy'] > d['base_accuracy'])}"
          f"/{len(d['runs'])} seeds above base",
          "training is unstable: lower lr, check gradient clipping")
    v.add("GP.3 seed noise", d["sigma_w"] <= T["sigma_w_max"],
          f"sigma_w={d['sigma_w']:.4f} (need <= {T['sigma_w_max']})",
          "raise the seed count; recompute every downstream MDE from this value")
    drops = [r["own_supervision_loss_drop"] for r in d["runs"]
             if np.isfinite(r["own_supervision_loss_drop"])]
    md = float(np.median(drops)) if drops else float("nan")
    v.add("GP.4 fits its own supervision", md >= T["own_supervision_loss_drop_min"],
          f"median loss drop {md:.2f} (need >= {T['own_supervision_loss_drop_min']})",
          "the V0.7 exclusion rule cannot work if a healthy run does not clear it")
    c2 = d["identity_c2"]["c2"]
    v.add("GP.5 estimator sane at mu = d",
          T["c2_identity_lo"] <= c2 <= T["c2_identity_hi"],
          f"C2={c2:.3f} (need in [{T['c2_identity_lo']}, {T['c2_identity_hi']}])",
          "the estimator is wrong on real text even though G0 passed")
    v.add("GP.6 headroom over MDE",
          d["headroom_ratio"] >= T["headroom_min_ratio"],
          f"gain/MDE = {d['headroom_ratio']:.1f}x "
          f"(need >= {T['headroom_min_ratio']}x; MDE={d['suite_mde_points']:.2f} pts)",
          "the dynamic range is too small for a correlation study: stronger teacher, "
          "weaker student, or harder data")
    v.add("GP.7 suite resolution",
          d["suite_mde_points"] <= T["suite_mde_max_points"],
          f"MDE={d['suite_mde_points']:.2f} pts on n={d['suite_n']} "
          f"(need <= {T['suite_mde_max_points']})",
          "add suite components: the design cannot resolve the predicted effects")
    add_resources(v, d)
    return v


def gate_g2(tag: str = "g2_validity") -> Verdict:
    r"""Is the coverage estimate valid on real text -- and which estimator is?

    Restructured after the first real run, the same way G0 was and for the same
    reason: the wiring failed the stage while the registered decision rule said to
    narrow and continue.  Recorded in ``thresholds.json`` revision 5.

    Design doc §4.5 fixed the responses in advance, and all three failures map to
    one of them.  ``G2.3`` failing means "estimator B is reading style; drop B and
    C and use A alone, **unless G2.7 also fails, in which case stop**".  ``G2.1``
    failing ends "if it cannot be fixed, use A only".  ``G2.6`` failing says "use A
    only, remove B and C from the paper".  The single criterion whose failure the
    protocol made a stop condition -- ``G2.5``, no measurable coverage difference
    anywhere -- passed at +0.187, and ``G2.7`` passed at a coefficient of variation
    of 0.015.  So the registered outcome is: estimator A alone.

    What blocks the stage is therefore A's own behaviour, and the thresholds for it
    are borrowed rather than invented, because inventing one after seeing A's
    numbers is the thing this protocol exists to prevent:

    * the null and paraphrase bands are ``w1_c2_lo``/``w1_c2_hi`` = [0.90, 1.15],
      registered in this stage for the null calibration of the coverage estimate --
      both tests require the coefficient to come out at one;
    * the sensitivity bar is ``c2_spread_min_ratio`` = 1.5, registered in G3 for
      "the intervention moves C2", which is the same question asked of W5.

    **Identification did not survive as designed, and the paper must say so.**
    Criterion C5 asked for two estimators agreeing in rank.  No estimator qualifies
    in both worlds: in the synthetic world A agreed with B at 0.868 and with C at
    0.643; on real text it agrees with C at 0.800 and with B at 0.600.  What stands
    in its place is stronger where it is available and weaker where it is not --
    A is validated against an exactly computable ground truth in G0 (rank 0.770,
    accuracy 0.93 in the comparable range, relative error falling 0.057, 0.016,
    0.005 with sample size), and on real text by four behavioural checks it has no
    way to pass by accident: null 1.05, order-shuffle 1.06, paraphrase 1.01, real
    shift 2.84, stable to 0.015 under re-binning.  Cross-estimator agreement was a
    proxy for ground truth; here the ground truth itself exists.  That is the
    accurate statement, and it is not the same as "C5 satisfied".
    """
    v = Verdict("G2 -- coverage estimate valid on real text?")
    d = runio.load(tag, where="results")
    if d is None:
        v.add("G2", None, f"results/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["g2"]
    T3 = runio.thresholds()["g3"]
    tests = d["tests"]

    def A(name: str, field: str = "c2_A"):
        return tests.get(name, {}).get(field, float("nan"))

    lo, hi = T["w1_c2_lo"], T["w1_c2_hi"]
    sens = T3["c2_spread_min_ratio"]

    # ---- the two registered stop conditions
    v.add("G2.5 excess over shortcut controls (stop condition)",
          tests.get("W5_real_shift", {}).get("excess_auc", float("nan"))
          >= T["excess_auc_min"],
          f"excess AUC={tests.get('W5_real_shift', {}).get('excess_auc', float('nan')):+.3f} "
          f"(need >= {T['excess_auc_min']})",
          "STOP: no pair has a measurable coverage difference, so there is nothing "
          "for the method to fix")
    acv = d["abstraction_cv"]["median"]
    v.add("G2.7 abstraction robustness (stop condition with G2.3)",
          acv <= T["abstraction_cv_max"],
          f"median CV across magnitude binnings {d['abstraction_cv']['settings']} "
          f"= {acv:.4f} (need <= {T['abstraction_cv_max']})",
          "STOP: with estimator B dropped and the abstraction unstable, nothing "
          "measures coverage")

    # ---- estimator A, the one every GPU stage computes
    v.add("G2.A1 A null calibration", lo <= A("W1_null") <= hi,
          f"C2(A)={A('W1_null'):.3f} on two halves of one measure "
          f"(need in [{lo}, {hi}])",
          "STOP: the load-bearing estimator does not read 1 where the answer is 1")
    v.add("G2.A2 A paraphrase invariance", lo <= A("W3_paraphrase") <= hi,
          f"C2(A)={A('W3_paraphrase'):.3f} after rewriting the wording and holding "
          f"the numeric states fixed (need in [{lo}, {hi}]); "
          f"discard rate {d['paraphrase_discard_rate']:.2f}, kept "
          f"{d['n_paraphrase_kept']}",
          "STOP: even the discrete abstraction is reading wording")
    v.add("G2.A3 A sensitivity to a real shift", A("W5_real_shift") >= sens,
          f"C2(A)={A('W5_real_shift'):.3f} (need >= {sens})",
          "STOP: A cannot see a coverage difference that is really there")
    sep = A("W5_real_shift") / max(A("W1_null"), 1e-9)
    v.add("G2.A4 A separates a real shift from the null", sep >= sens,
          f"W5/W1 = {sep:.2f}x (need >= {sens}x)",
          "STOP: A's response to a real shift is not distinguishable from noise")

    # ---- estimator B and C: diagnostics, and the registered consequence
    b_ok = {"W1 null": T["w1_auc_lo"] <= tests.get("W1_null", {}).get("auc_B", 9)
            <= T["w1_auc_hi"],
            "W2 sensitivity": tests.get("W2_shuffle", {}).get("auc_B", 0)
            >= T["w2_auc_min"],
            "W3 paraphrase": tests.get("W3_paraphrase", {}).get("auc_B", 9)
            <= T["w3_auc_max"]}
    v.add("G2.B estimator B: "
          + ("ADMISSIBLE" if all(b_ok.values()) else "DROPPED") + " [diagnostic]",
          None,
          ", ".join(f"{k}: {'ok' if ok else 'FAIL'}" for k, ok in b_ok.items())
          + f"  (W3 AUC={tests.get('W3_paraphrase', {}).get('auc_B', float('nan')):.3f} "
            f"vs bar {T['w3_auc_max']}; W1 AUC="
            f"{tests.get('W1_null', {}).get('auc_B', float('nan')):.3f} "
            f"vs band [{T['w1_auc_lo']}, {T['w1_auc_hi']}])"
          + ("" if all(b_ok.values()) else
             "  -- reads style, not state; dropped per design doc §4.5"))
    nc = d.get("w1_null_vs_n") or {}
    if nc:
        ns = sorted(nc, key=int)
        trend = ("falls toward 0.5 as n grows: under-sampled rather than wrong"
                 if nc[ns[-1]]["auc"] < nc[ns[0]]["auc"] - 0.05
                 else "flat in n: structurally wrong, not under-sampled")
        v.add("G2.1b W1 null vs sample size [diagnostic]", None,
              ", ".join(f"n={k}: AUC={nc[k]['auc']:.3f}" for k in ns)
              + f"  ({trend})")
    v.add("G2.4 W4 source invariance [not tested]", None,
          f"only {d['n_student_matched']} student traces matched a teacher trace's "
          f"numeric signature (need {T['w4_auc_min_pairs']}); the paper must report "
          f"this test as not run rather than as passed")
    v.add("G2.6 cross-estimator agreement [diagnostic]", None,
          ", ".join(f"{k}: rho={vv['rho']:.3f}" for k, vv in d["pairwise"].items())
          + f" (bar {T['estimator_pairwise_rho_min']}). No estimator agrees with A "
            f"in BOTH worlds, so criterion C5 is not met as designed; A rests on "
            f"ground-truth validation in G0 instead, and the paper must say so")
    add_resources(v, d)
    return v


def gate_g3(tag: str = "g3_measure_vs_loss") -> Verdict:
    v = Verdict("G3 -- objective or measure?")
    d = runio.load(tag, where="results")
    if d is None:
        v.add("G3", None, f"results/{tag}.json missing -- run the stage first")
        return v
    if d.get("partial"):
        # A worker checkpoints its blob after every arm so a kill loses nothing.
        # That copy holds one worker's share of the design, so its analysis would be
        # computed on a fraction of it.  Blocking, not diagnostic: a verdict read off
        # a fraction is worse than no verdict.
        v.add("G3 blob is one worker's mid-run checkpoint", False,
              f"results/{tag}.json was saved by a worker that had finished "
              f"{d.get('arms_this_worker', '?')} arms and had not ended its sweep",
              "run merge_g3.py to union the workers into the canonical blob, then "
              "gate that")
        return v
    T = runio.thresholds()["g3"]
    probe = d.get("capacity_probe") or {}
    if probe:
        spread = probe.get("recovery_spread_points", float("nan"))
        need = T.get("capacity_recovery_spread_min_points", 15.0)
        v.add("G3.0 student sweep spans the capacity axis",
              bool(probe.get("wide_enough")),
              f"recovery spans {spread:.1f} pts across students (need >= {need}); "
              + ", ".join(f"{k.split('/')[-1]}={100*vv['recovery']:.0f}%"
                          for k, vv in (probe.get("per_student") or {}).items()
                          if vv.get("recovery") == vv.get("recovery")),
              "the comparison cannot be interpreted: a null result would not "
              "separate a wrong theory from a wrong regime. Add a larger student "
              "(Qwen2.5-3B) or a stronger teacher and rerun")
    if d.get("aborted"):
        v.add("G3 main comparison", False, f"not run: {d['aborted']}",
              "fix the design precondition above, then rerun the stage")
        add_resources(v, d)
        return v

    v.add("V4.6 intervention moves C2",
          d["c2_spread_ratio"] >= T["c2_spread_min_ratio"],
          f"C2 spread {d['c2_spread_ratio']:.2f}x "
          f"(need >= {T['c2_spread_min_ratio']}x)",
          "the manipulation failed before training: redesign the mask, do not "
          "interpret the accuracies")
    for st, e in d["anova"].items():
        if not e:
            continue
        ratio = e["eta2_measure"] / max(e["eta2_loss"], 1e-9)
        v.add(f"G3.1 measure over loss [{st.split('/')[-1]}]",
              ratio >= T["eta2_measure_over_loss_min_ratio"],
              f"eta2 measure={e['eta2_measure']:.3f} vs loss={e['eta2_loss']:.3f} "
              f"(ratio {ratio:.2f}, need >= {T['eta2_measure_over_loss_min_ratio']})",
              "the objective dominates: the divergence literature is right and this "
              "direction is not -- STOP")
    mde = d["suite_mde_points"]
    for k, c in d["contrast"].items():
        eff = c["uniform_minus_worst_points"]
        v.add(f"G3.2 coverage effect [{k}]",
              eff >= max(T["unif_minus_worst_min_points"], mde)
              and c["seed_test"]["p"] <= 0.05,
              f"uniform - {c['worst_mode']} = {eff:+.2f} pts "
              f"(need >= {max(T['unif_minus_worst_min_points'], mde):.2f}), "
              f"p={c['seed_test']['p']:.3f}",
              "position distribution does not matter; STOP unless the interaction "
              "below is present")
    for obj, i in d.get("interaction", {}).items():
        v.add(f"G3.3 capacity interaction [{obj}]",
              i["interaction_points"] >= T["interaction_min_points"],
              f"effect {i['effect_points']} -> "
              f"{i['interaction_points']:+.2f} pts larger for the bigger student "
              f"(need >= {T['interaction_min_points']})",
              "the synthetic gate predicted coverage matters more at ample "
              "capacity; if this is absent, the regime account is wrong")
    for st, b in d["base"].items():
        floor = max((r["accuracy"] for r in d["runs"]
                     if r["student"] == st and r["mode"] == "uniform"), default=0)
        v.add(f"V0.1 floor beats base [{st.split('/')[-1]}]",
              100 * (floor - b["accuracy"]) >= T["floor_beats_base_min_points"],
              f"{100*(floor-b['accuracy']):+.2f} pts "
              f"(need >= {T['floor_beats_base_min_points']})",
              "this configuration has no signal; fix it and discard these numbers")
    add_resources(v, d)
    return v


def gate_g4(tag: str = "g4_predict") -> Verdict:
    v = Verdict("G4 -- does C2 predict capability across pairs?")
    d = runio.load(tag, where="results")
    if d is None:
        v.add("G4", None, f"results/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["g4"]
    rel = d["reliability"].get("reliability", float("nan"))
    v.add("V5.7 dependent-variable reliability", rel >= T["reliability_min"],
          f"reliability={rel:.3f} (need >= {T['reliability_min']}), "
          f"design ceiling={d['disattenuated']['design_ceiling']:.3f}",
          "the grid is too homogeneous: a low correlation here would be a property "
          "of the design, not of the theory. Widen the grid before concluding")
    v.add("V5.1 C2 spread", d["c2_spread_ratio"] >= T["c2_spread_min_ratio"],
          f"max/min={d['c2_spread_ratio']:.1f} "
          f"(need >= {T['c2_spread_min_ratio']})",
          "a correlation on a narrow grid means nothing")
    v.add("V5.2 predictors not collinear",
          abs(d["collinearity_c2_fid"]["rho"]) <= T["predictor_collinearity_max_rho"],
          f"rho(C2, fidelity)={d['collinearity_c2_fid']['rho']:+.3f}",
          "the two factors cannot be separated in this grid; add a measure mode")
    v.add("G4.5 mostly positive gains",
          d["positive_gain_fraction"] >= T["positive_gain_min_frac"],
          f"{d['positive_gain_fraction']:.2f} of points gained "
          f"(need >= {T['positive_gain_min_frac']})",
          "several configurations have no signal; report which were dropped and why")
    v.add("G4.1 product predicts",
          d["rho_product"]["rho"] >= T["rho_product_min"],
          f"rho={d['rho_product']['rho']:+.3f} (need >= {T['rho_product_min']}); "
          f"corrected {d['disattenuated']['rho_corrected']:+.3f}",
          "publish as a negative result: a bound that holds and does not transfer")
    v.add("G4.2 fidelity alone does not",
          d["rho_fidelity"]["rho"] <= T["rho_fidelity_max"],
          f"rho={d['rho_fidelity']['rho']:+.3f} (need <= {T['rho_fidelity_max']})",
          "fidelity already predicts; the coverage term adds nothing here")
    v.add("G4.3 separation significant",
          d["rho_gap"] >= T["rho_gap_min"] and d["williams"]["p"] <= T["williams_max_p"],
          f"gap={d['rho_gap']:+.3f}, Williams p={d['williams']['p']:.4f}",
          "report the two correlations separately instead of their difference")
    pr = d.get("partial_regression")
    if pr and "log_cov" in pr["names"]:
        i = pr["names"].index("log_cov")
        v.add("G4.4 coverage term after controls",
              pr["p"][i] <= T["partial_beta_cov_max_p"],
              f"beta_cov={pr['beta'][i]:+.3f}, p={pr['p'][i]:.2e} "
              f"(controls: fidelity, Gamma)",
              "the effect is explained by the capability gap alone")
    return v


def gate_g5(tag: str = "g5_main") -> Verdict:
    v = Verdict("G5 -- method against baselines at matched teacher calls")
    d = runio.load(tag, where="results")
    if d is None:
        v.add("G5", None, f"results/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["g5"]
    mde = d["suite_mde_points"]
    for name, need, action in (
            ("vs_floor", T["vs_floor_min_points"],
             "the method does not beat its own degeneracy floor"),
            ("vs_best_baseline", T["vs_best_baseline_min_points"],
             "publish as an orthogonal axis and lean on the reverse-KL composition arm"),
            ("vs_random_prefix", T["vs_random_prefix_min_points"],
             "SELECTION CARRIES NO INFORMATION. Read G6.1: if the selection rule did "
             "lower C2 and this arm still ties, the theory is refuted; if it did not, "
             "the intervention failed and the theory is untested")):
        k = d["key_tests"].get(name, {})
        diff = k.get("median_diff_points", float("nan"))
        ps = k.get("p_values", [])
        p = float(np.median(ps)) if ps else float("nan")
        v.add(f"G5 {name}", diff >= max(need, mde) and p <= T["max_p"],
              f"{diff:+.2f} pts (need >= {max(need, mde):.2f}), median p={p:.4f}",
              action)
    signs = []
    for cfg, tests in d["per_config_tests"].items():
        if "vs_floor" in tests:
            signs.append(np.sign(tests["vs_floor"]["diff"]))
    v.add("G5.6 same sign everywhere",
          len(signs) >= T["same_sign_configs_required"] and len(set(signs)) == 1,
          f"{len(signs)} configurations, signs {sorted(set(signs))}",
          "the result is configuration-specific; do not state it generally")
    disp = {}
    for r in d["runs"]:
        disp.setdefault(r["arm"], []).append(r.get("displacement", float("nan")))
    ref = np.nanmedian(disp.get("sft", [np.nan]))
    if np.isfinite(ref) and ref > 0:
        m = np.nanmedian(disp.get("scd2", [np.nan]))
        v.add("V0.3 comparable displacement",
              T["displacement_ratio_lo"] <= m / ref <= T["displacement_ratio_hi"],
              f"||dtheta|| ratio scd2/sft = {m/ref:.2f}",
              "the arms moved by different amounts and have not been compared")
    add_resources(v, d)
    return v


def gate_g6(tag: str = "g6_mechanism") -> Verdict:
    v = Verdict("G6 -- does the selection rule lower C2 out of sample?")
    d = runio.load(tag, where="results")
    if d is None:
        v.add("G6", None, f"results/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["g6"]
    r = d["ratio_over_random"]
    cc = d.get("cell_coverage")
    if cc is not None:
        T6 = runio.thresholds()["g6"]
        lo = T6.get("min_cell_coverage", 0.20)
        hi = T6.get("max_cell_coverage", 0.80)
        v.add("G6.0 measure is partially deficient", bool(lo <= cc <= hi),
              f"supervision covers {cc:.1%} of the cells the teacher visits "
              f"(need {lo:.0%}-{hi:.0%})",
              "the comparison cannot come out: a measure covering almost nothing "
              "is fixed by any diverse addition, and one covering almost "
              "everything has nothing to fix. Adjust --keep-cells and rerun")
    if d.get("aborted"):
        v.add("G6 branch sweep", False, f"not run: {d['aborted']}",
              "fix the precondition above, then rerun the stage")
        add_resources(v, d)
        return v

    sweep = d.get("sweep") or {}
    cp = d.get("coupon_threshold")
    if sweep and cp:
        T6 = runio.thresholds()["g6"]
        bmax = max(int(b) for b in sweep)
        v.add("G6.0b budget is scarce relative to the gap",
              bmax <= T6.get("max_budget_frac", 0.5) * cp,
              f"largest budget {bmax} vs coupon-collector threshold {cp:.0f} "
              f"({bmax / cp:.2f}x; need <= "
              f"{T6.get('max_budget_frac', 0.5)})",
              "random selection saturates at this budget, so no rule can "
              "distinguish itself: lower the budgets and rerun")
        v.add("G6.1b best rule beats random at some budget [diagnostic]", None,
              ", ".join(f"{b}: {sv['best_rule']} {sv['best_over_random']:.3f}x"
                        for b, sv in sorted(sweep.items(), key=lambda x: int(x[0]))))
    v.add("G6.1 ratio beats random", r >= T["c2_reduction_advantage_min_ratio"],
          f"held-out C2 reduction ratio = {r:.2f} "
          f"(need >= {T['c2_reduction_advantage_min_ratio']})",
          f"try the alternatives (best observed: {d.get('best_rule')}); if none "
          f"clears this, no intervention lowers C2 and the direction stops here")
    return v


def gate_g7(tag: str = "g7_dissociation") -> Verdict:
    v = Verdict("G7 -- prompt coverage or state coverage?")
    d = runio.load(tag, where="results")
    if d is None:
        v.add("G7", None, f"results/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["g7"]
    v.add("V8 arms reach many cells",
          d["min_distinct_cells"] >= T["min_distinct_sigma_states"],
          f"smallest arm reaches {d['min_distinct_cells']} cells "
          f"(need >= {T['min_distinct_sigma_states']})",
          "the one-prompt arm has no coverage; the comparison is empty")
    v.add("G7.1 states dominate prompts",
          d["beta_ratio_cells_over_prompts"] >= T["beta_prefix_over_prompt_min_ratio"],
          f"|beta_cells|/|beta_prompts| = "
          f"{d['beta_ratio_cells_over_prompts']:.2f} "
          f"(need >= {T['beta_prefix_over_prompt_min_ratio']})",
          "task coverage is the mechanism, not state coverage: the theory must be "
          "restated over prompts, which is a different paper")
    return v


def gate_g8(tag: str = "g8_cost") -> Verdict:
    v = Verdict("G8 -- specialisation cost and reachable set")
    d = runio.load(tag, where="results")
    if d is None:
        v.add("G8", None, f"results/{tag}.json missing -- run the stage first")
        return v
    T = runio.thresholds()["g8"]
    ratio = d["general_loss_ratio_vs_fkl"].get("scd2", float("nan"))
    v.add("G8.1 keeps general ability",
          ratio <= T["general_ability_loss_max_ratio_vs_fkl"],
          f"general-ability loss relative to forward KL = {ratio:.2f} "
          f"(need <= {T['general_ability_loss_max_ratio_vs_fkl']})",
          "report that the method does not reduce specialisation cost")
    v.add("G8.2 reachable set expanded",
          bool(d["passk_expanded"].get("scd2", False)),
          f"pass@64 above base: {d['passk_expanded']}",
          "the method reallocates probability inside the existing solution set")
    return v


REGISTRY = {"gp": gate_gp, "g2": gate_g2, "g3": gate_g3, "g4": gate_g4,
            "g5": gate_g5, "g6": gate_g6, "g7": gate_g7, "g8": gate_g8}
