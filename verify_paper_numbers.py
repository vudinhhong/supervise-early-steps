#!/usr/bin/env python
"""Every number the conference paper states, recomputed from the result files.

Run:  python verify_paper_numbers.py

Each line prints the claim as the paper makes it, the value recomputed from the
blob next to it, and OK or MISMATCH.  The point is that the mapping from paper to
evidence is executable rather than asserted: if a number here ever stops matching,
the paper is wrong and this script says so.
"""
import json, pathlib, re, collections
import numpy as np

R = pathlib.Path(__file__).resolve().parent / "results"
S05, S15 = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"
ok_count, bad_count = 0, 0


def load(n):
    return json.loads((R / n).read_text())


def check(label, claimed, got, tol=None):
    global ok_count, bad_count
    if isinstance(claimed, str) or isinstance(got, str):
        good = str(claimed) == str(got)
    else:
        # Tolerance follows the precision the paper prints: a value written to
        # three decimals must agree to half a unit of the last digit.  The old
        # rule, tol * max(1, |claimed|), allowed 0.02 absolute on every number
        # below one, which is four times a quantity such as 0.005.
        c = float(claimed)
        if tol is None:
            txt = ("%g" % c)
            dec = len(txt.split(".")[1]) if "." in txt else 0
            lim = 0.5 * 10 ** (-dec)
        else:
            lim = tol * max(abs(c), 1e-12)
        good = abs(c - float(got)) <= lim * (1.0 + 1e-9) + 1e-12
    ok_count += good
    bad_count += (not good)
    print("  %-52s expect=%-12s file=%-12s %s"
          % (label, claimed, (("%.4g" % got) if not isinstance(got, str) else got),
             "OK" if good else "MISMATCH"))


# ---------------------------------------------------------------------------
# SECTION 0: the manuscript's own text.
#
# Every other check in this file compares a blob with a constant typed here, so a
# value that goes stale in conf.tex while the constant is updated still passes.
# This section closes that gap for the numbers the manuscript prints outside the
# tables: it recomputes them from the blobs, renders them the way the manuscript
# does, and requires the string to appear in conf.tex.
# ---------------------------------------------------------------------------
TEX = next((c for c in (R.parent / "conf.tex", R.parent.parent / "conf.tex")
            if c.exists()), None)
TEXT = " ".join(TEX.read_text().split()) if TEX else None


def in_tex(label, snippet):
    if TEXT is None:
        return
    check("conf.tex: " + label, True, snippet in TEXT)


print("=" * 100)
print("SECTION 0  what is checked against what")
print("=" * 100)
print("  Every value below is recomputed from results/ and compared with the")
print("  number this script expects. Many of them the manuscript no longer")
print("  prints in running text: they are what Figures 1 and 2 plot, and they")
print("  are checked here because a stale figure is as wrong as a stale number.")
if TEXT is None:
    print("  conf.tex is not beside the package, so the manuscript text itself")
    print("  is unchecked; with it present, Section 8 requires the manuscript to")
    print("  print the strings recomputed here.")
else:
    print("  conf.tex is beside the package, so Section 8 also requires the")
    print("  manuscript to print the strings recomputed here.")

print()
print("=" * 100)
print("SECTION 6.1  positive control and the exactly solvable world")
print("=" * 100)
gp = load("gp_control.json")
check("base accuracy of Qwen2.5-0.5B (%)", 13.94, 100 * gp["base_accuracy"])
check("gain after distillation (points)", 30.87, gp["gain_points"])
check("teacher accuracy (%)", 79.52, 100 * gp["teacher_accuracy"])
check("available gap Gamma (points)", 65.57, gp["gamma_points"])
check("share of Gamma recovered", 0.471, gp["private_value_recovery"])
check("seed-to-seed sd (points)", 0.32, 100 * gp["sigma_w"])
check("all three seeds positive", "True", str(gp["all_seeds_positive"]))
check("evaluation suite size", 3471, gp["suite_n"])
check("suite MDE (points)", 2.13, gp["suite_mde_points"])

g0 = load("g0_estimator.json")
check("configurations swept", 780, g0["world_validity"]["n_configs"])
check("violations of Theorem 1", 0, g0["world_validity"]["theorem1_violations"])
check("rho(C2, E_mu[KL])", -0.217, g0["world_validity"]["rho_c2_ekl"])
check("dissociating configurations", 46, g0["world_validity"]["n_dissociating"])
for n, want in (("1000", 0.057), ("10000", 0.016), ("100000", 0.005)):
    check("estimator A median rel. error at n=%s" % n, want,
          g0["convergence"]["c2_A"][n])
check("estimator A median CV", 0.014, g0["stability_cv"]["c2_A"]["median_cv"])
check("estimator A respects the sigma lower bound", 0.998,
      g0["lower_bound"]["A_le_exact_sigma"])
check("estimator B median rel. error at n=1e5", 0.026,
      g0["convergence"]["c2_B"]["100000"])
check("estimator C median rel. error at n=1e5", 0.821,
      g0["convergence"]["c2_C"]["100000"])

# Corollary 1, with the registered criterion: raw loss <= 1e-6 AND gap >= 0.10
ex = g0["exact"]
inst = [r for r in ex if r["e_kl_mu_raw"] <= 1e-6 and r["gap"] >= 0.10]
check("Corollary 1 instances (loss<=1e-6, gap>=0.10)", 47, len(inst))
check("largest gap among them", 0.523, max(r["gap"] for r in inst))

print()
print("=" * 100)
print("SECTION 6.2  behavioural validity on real teacher traces")
print("=" * 100)
g2 = load("g2_validity.json")
for test, c2, auc, exc in (("W1_null", 1.047, 0.544, 0.042),
                           ("W2_shuffle", 1.056, 0.9998, 0.475),
                           ("W3_paraphrase", 1.009, 0.656, 0.143),
                           ("W5_real_shift", 2.842, 0.976, 0.187)):
    tv = g2["tests"][test]
    check("%-14s C2 of estimator A" % test, c2, tv["c2_A"])
    check("%-14s AUC of estimator B" % test, auc, tv["auc_B"])
    check("%-14s excess AUC over controls" % test, exc, tv["excess_auc"])
check("abstraction CV under re-binning", 0.015, g2["abstraction_cv"]["median"])
check("W4 matched rows (inconclusive)", 17, g2["tests"]["W4_source"]["n_rows_b"])

print()
print("=" * 100)
print("SECTION 6.3  the factorial: supervision distribution versus objective")
print("=" * 100)
g3 = load("g3_measure_vs_loss.json")
check("arms completed", 80, len(g3["runs"]))
check("supervised steps matched to", 37509, g3["matched_count"])
check("C2 spread across masks (x)", 38.5, g3["c2_spread_ratio"])
for m, want in (("uniform", 1.017), ("early", 5.360), ("late", 14.469),
                ("cluster_drop", 39.148)):
    check("C2 of mask %-13s" % m, want, g3["coverage"][m]["c2"])
for st, lab in ((S05, "0.5B"), (S15, "1.5B")):
    check("base accuracy %s (%%)" % lab,
          6.66 if st == S05 else 3.23, 100 * g3["base"][st]["accuracy"])
cells = {(S05, "ce"): [41.64, 39.83, 24.97, 4.48],
         (S05, "fkl"): [43.01, 40.60, 32.38, 30.50],
         (S15, "ce"): [65.62, 65.46, 44.28, 5.94],
         (S15, "fkl"): [66.25, 65.40, 48.10, 59.98]}
for (st, ob), want in cells.items():
    for m, w in zip(("uniform", "early", "late", "cluster_drop"), want):
        v = [100 * r["accuracy"] for r in g3["runs"]
             if r["student"] == st and r["objective"] == ob and r["mode"] == m]
        check("accuracy %s/%s/%s (%%)" % (st.split("/")[-1], ob, m), w,
              float(np.mean(v)), tol=0.005)
# the standard deviations Figure 1 plots as seed clouds
sds = {(S05, "ce"): [0.44, 0.56, 6.10, 2.06],
       (S05, "fkl"): [0.36, 0.43, 3.50, 3.27],
       (S15, "ce"): [0.43, 0.50, 8.16, 4.64],
       (S15, "fkl"): [0.32, 0.25, 13.78, 3.01]}
for (st, ob), want in sds.items():
    for m, w in zip(("uniform", "early", "late", "cluster_drop"), want):
        v = np.array([100 * r["accuracy"] for r in g3["runs"]
                      if r["student"] == st and r["objective"] == ob and r["mode"] == m])
        check("sd %s/%s/%s" % (st.split("/")[-1], ob, m), w, float(v.std(ddof=1)))

for st, lab, wm, wo, wi in ((S05, "0.5B", 0.653, 0.131, 0.173),
                            (S15, "1.5B", 0.475, 0.132, 0.322)):
    e = g3["anova"][st]
    check("eta2 measure   %s" % lab, wm, e["eta2_measure"])
    check("eta2 objective %s" % lab, wo, e["eta2_loss"])
    check("eta2 interact. %s" % lab, wi, e["eta2_interaction"])
    check("ratio measure/objective %s" % lab, 5.00 if st == S05 else 3.60,
          e["eta2_measure"] / e["eta2_loss"])
check("design reliability", 0.989, g3["reliability"]["reliability"])
for key, want, lo, hi in (("Qwen2p5-0p5B|ce", 37.17, 35.37, 38.63),
                          ("Qwen2p5-0p5B|fkl", 12.50, 9.95, 15.06),
                          ("Qwen2p5-1p5B|ce", 59.68, 55.44, 62.51),
                          ("Qwen2p5-1p5B|fkl", 18.15, 8.52, 30.34)):
    c = g3["contrast"][key]
    check("best-minus-worst mask %s (points)" % key, want,
          c["uniform_minus_worst_points"])
    check("  interval low  %s" % key, lo, 100 * c["seed_test"]["lo"])
    check("  interval high %s" % key, hi, 100 * c["seed_test"]["hi"])

print()
print("=" * 100)
print("SECTION 6.3  the supervision-quantity control")
print("=" * 100)
tok = {}
for r in g3["runs"]:
    tok.setdefault(r["mode"], set()).add(r["n_supervised_positions"])
for m, want in (("uniform", 591638), ("early", 681650), ("late", 456821),
                ("cluster_drop", 146055)):
    check("supervised token positions, mask %-13s" % m, want,
          sorted(tok[m])[0], tol=0.001)
g7 = load("g7_dissociation.json")
by = collections.defaultdict(list)
for r in g7["runs"]:
    by[(r["prompts"], r["budget"])].append(r)
print("  -- at matched prompt count, quadrupling token positions:")
for p in (100, 1000, 6000):
    a, b = by[(p, 25000)], by[(p, 100000)]
    ta = np.mean([x["n_supervised_positions"] for x in a])
    tb = np.mean([x["n_supervised_positions"] for x in b])
    da = 100 * (np.mean([x["accuracy"] for x in b])
                - np.mean([x["accuracy"] for x in a]))
    print("     prompts=%-5d %8d -> %8d tokens (%.2fx)   accuracy %+.2f points"
          % (p, ta, tb, tb / ta, da))
one = by[(1, 25000)]
check("the 38K-position configuration: tokens", 38091,
      np.mean([x["n_supervised_positions"] for x in one]), tol=0.001)
check("the 38K-position configuration: accuracy (%)", 65.52,
      100 * np.mean([x["accuracy"] for x in one]))

print()
print("=" * 100)
print("SECTION 6.5  the selection rule does not pay")
print("=" * 100)
g6 = load("g6_mechanism.json")
check("abstract states the teacher visits", 413,
      g6["before"]["selection"]["n_states_seen_d"])
check("states the supervision touches", 202,
      g6["before"]["selection"]["n_states_seen_mu"])
check("C2 on the selection pool", 23.3, g6["before"]["selection"]["c2"])
check("C2 on the held-out pool", 22.3, g6["before"]["heldout"]["c2"])
check("best rule over random, best budget (x)", 1.0235, g6["best_over_random"])
check("ratio rule over random, averaged (x)", 1.0126, g6["ratio_over_random"])

print()
print("=" * 100)
print("SECTION 6.3b  the problem-diversity audit")
print("=" * 100)
# Parsed from the captured output of audit/audit_rows.py and audit/audit_earlylate.py,
# which need the trace corpus and so run on the training host, not here.
A = pathlib.Path(__file__).resolve().parent / "audit"
rows = {}
for ln in (A / "audit_rows.txt").read_text().splitlines():
    f = ln.split()
    if len(f) == 7 and f[0] in ("uniform", "early", "late", "cluster_drop"):
        rows[f[0]] = dict(n=int(f[1]), share=float(f[2].rstrip("%")),
                          steps=int(f[3]), per_row=float(f[4]))
for m in ("uniform", "early", "late"):
    check("rows entering training, %s" % m, 15831, rows[m]["n"], tol=0.0)
    check("share of corpus, %s (%%)" % m, 93.4, rows[m]["share"])
    check("supervised steps/row, %s" % m, 2.37, rows[m]["per_row"])
check("rows entering training, cluster_drop", 6634, rows["cluster_drop"]["n"], tol=0.0)
check("share of corpus, cluster_drop (%)", 39.1, rows["cluster_drop"]["share"])
check("supervised steps/row, cluster_drop", 5.65, rows["cluster_drop"]["per_row"])
check("problem-diversity ratio (x)", 2.4,
      rows["uniform"]["n"] / rows["cluster_drop"]["n"])
for m in rows:
    check("supervised steps matched, %s" % m, 37509, rows[m]["steps"], tol=0.0)

el = (A / "audit_earlylate.txt").read_text()
check("early/late identical row sets", 1.0,
      1.0 if "early == late ?  True" in el else 0.0, tol=0.0)
for m in ("early", "late"):
    g = re.search(r"%s\s+mean trace length=([\d.]+)\s+mean supervised steps=([\d.]+)"
                  r"\s+ratio=([\d.]+)\s+corr\(len,sup\)=\+([\d.]+)" % m, el)
    check("mean trace length, %s" % m, 11.67, float(g.group(1)))
    check("mean supervised steps, %s" % m, 2.37, float(g.group(2)))
    check("supervision-to-length ratio, %s" % m, 0.216, float(g.group(3)))
    check("corr(length, supervised), %s" % m, 0.893, float(g.group(4)))
g = re.search(r"early (\d+) tokens \| late (\d+) tokens \| ratio ([\d.]+)x", el)
check("early supervised token positions", 681650, int(g.group(1)), tol=0.0)
check("late supervised token positions", 456821, int(g.group(2)), tol=0.0)
check("early/late token ratio (x)", 1.49, float(g.group(3)))
for m in ("uniform", "early", "late"):
    g = re.search(r"%s\s+(\d+)/\s*(\d+) rows with logit cache \(([\d.]+)%%\)" % m, el)
    check("teacher logit-cache coverage, %s (%%)" % m, 99.9, float(g.group(3)))

g3b = load("g3_measure_vs_loss.json")
ce15 = lambda mode: 100 * np.mean(
    [r["accuracy"] for r in g3b["runs"] if r["mode"] == mode
     and r["objective"] == "ce" and r["student"].endswith("1.5B")])
check("the clean early-vs-late gap (points)", 21.18, ce15("early") - ce15("late"))

# the variance ratio refitted on the three masks with identical row sets
def eta2(student, modes):
    r = [x for x in g3b["runs"] if x["student"].endswith(student)
         and x["mode"] in modes]
    y = np.array([x["accuracy"] for x in r]); gm = y.mean()
    sst = ((y - gm) ** 2).sum()
    ss = {}
    for fac in ("mode", "objective"):
        ss[fac] = sum(
            len(v) * (np.mean(v) - gm) ** 2
            for v in ([x["accuracy"] for x in r if x[fac] == lev]
                      for lev in sorted({x[fac] for x in r})))
    return ss["mode"] / sst, ss["objective"] / sst

FOUR = {"uniform", "early", "late", "cluster_drop"}
THREE = {"uniform", "early", "late"}
for stud, em4, el4, ratio4 in (("0.5B", 0.6534, 0.1307, 5.00),
                               ("1.5B", 0.4754, 0.1319, 3.60)):
    em, el = eta2(stud, FOUR)
    check("eta2 measure, four masks, %s" % stud, em4, em)
    check("eta2 objective, four masks, %s" % stud, el4, el)
    check("variance ratio, four masks, %s" % stud, ratio4, em / el)
for stud, em3, el3, ratio3 in (("0.5B", 0.758, 0.0533, 14.2),
                               ("1.5B", 0.704, 0.0045, 156.6)):
    em, el = eta2(stud, THREE)
    check("eta2 measure, cluster-drop excluded, %s" % stud, em3, em)
    check("eta2 objective, cluster-drop excluded, %s" % stud, el3, el, tol=0.1)
    check("variance ratio, cluster-drop excluded, %s" % stud, ratio3, em / el,
          tol=0.1)

print()
print("=" * 100)
print("SECTION 6.3b  the matched pair and the baselines")
print("=" * 100)

# seed-level MDE across the four uniform cells, two-arm form 2.8 s sqrt(2/n)
_g3 = load("g3_measure_vs_loss.json")
def _cell(st, obj, mode):
    return np.array([100 * r["accuracy"] for r in _g3["runs"]
                     if r["student"] == st and r["objective"] == obj
                     and r["mode"] == mode])
mdes = []
for st in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
    for obj in ("ce", "fkl"):
        u = _cell(st, obj, "uniform")
        mdes.append(2.8 * u.std(ddof=1) * np.sqrt(2.0 / len(u)))
check("seed-level MDE, lowest uniform cell", 0.57, float(min(mdes)))
check("seed-level MDE, highest uniform cell", 0.78, float(max(mdes)))
gpc = load("gp_control.json")
check("seed-level MDE, positive control (single arm)", 0.52, gpc["seed_mde_points"])
def t_quantile(df, q=0.975):
    """Inverse Student t by bisection on a numerically integrated cdf.

    The intervals in the paper are Welch--Satterthwaite, so the degrees of
    freedom are fractional and differ per cell; a single hard-coded constant
    cannot serve.  No SciPy, to keep the package dependency-free.
    """
    xs = np.linspace(0.0, 60.0, 600001)
    pdf = (1.0 + xs * xs / df) ** (-(df + 1.0) / 2.0)
    cdf = np.concatenate(([0.0], np.cumsum((pdf[1:] + pdf[:-1]) * 0.5 * (xs[1] - xs[0]))))
    half = cdf / (2.0 * cdf[-1])          # P(0 <= T <= x) normalised
    return float(np.interp(q - 0.5, half, xs))


def welch(a, b):
    """Mean difference, its standard error and the Welch degrees of freedom."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(a.mean() - b.mean()), se, float(df)

g3r = load("g3_measure_vs_loss.json")
S05, S15 = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"

def cell(st, obj, mode):
    return np.array([100 * r["accuracy"] for r in g3r["runs"]
                     if r["student"] == st and r["objective"] == obj
                     and r["mode"] == mode])

# the four early-late cells, as Welch--Satterthwaite intervals over five seeds
for st, nm, obj, gap, lo, hi in (
        (S05, "0.5B", "ce",  14.87,  7.30, 22.43),
        (S05, "0.5B", "fkl",  8.22,  3.89, 12.54),
        (S15, "1.5B", "ce",  21.18, 11.06, 31.31),
        (S15, "1.5B", "fkl", 17.30,  0.20, 34.41)):
    e, l = cell(st, obj, "early"), cell(st, obj, "late")
    d, se, df = welch(e, l)
    t = t_quantile(df)
    check("early-late %s %s: difference" % (nm, obj), gap, d)
    check("early-late %s %s: interval low" % (nm, obj), lo, d - t * se)
    check("early-late %s %s: interval high" % (nm, obj), hi, d + t * se)

# soft targets on the matched arm: both intervals contain zero
for st, nm, want, lo, hi in ((S05, "0.5B", 7.42, -0.18, 15.01),
                             (S15, "1.5B", 3.83, -13.38, 21.03)):
    ce_, fk = cell(st, "ce", "late"), cell(st, "fkl", "late")
    d, se, df = welch(fk, ce_)
    t = t_quantile(df)
    check("soft-target gain on late, %s" % nm, want, d)
    check("soft-target interval low, %s" % nm, lo, d - t * se)
    check("soft-target interval high, %s" % nm, hi, d + t * se)
    check("soft-target interval contains zero, %s" % nm, 1.0,
          1.0 if (d - t * se) < 0 < (d + t * se) else 0.0, tol=0.0)

# the per-seed values of late printed in the table
for st, nm, obj, want in ((S05, "0.5B", "ce",  [14.3, 26.6, 26.8, 27.4, 29.8]),
                          (S05, "0.5B", "fkl", [26.65, 32.35, 33.25, 33.54, 36.13]),
                          (S15, "1.5B", "ce",  [34.3, 37.8, 46.7, 48.1, 54.5]),
                          (S15, "1.5B", "fkl", [25.8, 48.9, 49.4, 53.1, 63.4])):
    got = sorted(cell(st, obj, "late"))
    for k, (w, g) in enumerate(zip(want, got)):
        check("late per-seed %s %s [%d]" % (nm, obj, k), w, g)


# the base measured two ways, and the format-learning headroom between them
gp = load("gp_control.json")
check("positive-control base, four-shot (%)", 13.94, 100 * gp["base_accuracy"])
check("shots used for that baseline", 4, gp["base_shots_used"], tol=0)
check("zero-shot base of the same model (%)", 6.57,
      100 * gp["base_references"]["zero_shot_chat"]["accuracy"])
check("format-learning headroom recorded by the control (points)", 7.375,
      gp["format_learning_headroom_points"])

# the bound's looseness on the exactly solvable world
g0r = load("g0_estimator.json") if (R / "g0_estimator.json").exists() else None
if g0r:
    ex = g0r["exact"]
    gg = np.array([e["gap"] for e in ex]); bb = np.array([e["bound"] for e in ex])
    check("violations of Theorem 1 (tol 1e-9)", 0,
          int(sum(1 for e in ex if e["gap"] > e["bound"] + 1e-9)), tol=0)
    t = bb / np.maximum(gg, 1e-12)
    check("median bound/gap", 232, float(np.median(t[np.isfinite(t)])), tol=0.03)
    check("configurations with bound < 1 (%)", 12.3, 100 * float((bb < 1).mean()))
else:
    print("  (g0_estimator.json not in this package -- skipped the tightness checks)")

# the ceiling of eq. (5), which the paper now states
cl = R.parent / "audit" / "ceiling.txt"
if cl.exists():
    import re as _re
    txt = cl.read_text()
    # Recompute the right-hand side of Eq. (5) from the raw figures in the file,
    # rather than reading a printed ratio: the file prints the naive max/E, while
    # the paper quotes (max w^2 - C2) / (E_d[w^2] - C2).
    rows = _re.findall(
        r"(selection|heldout)\s+(unclipped|clipped[^ ]*[^=]*?)\s+E_d\[w\^2\]=\s*([\d.]+)"
        r"\s+max w\^2=\s*([\d.]+)", txt)
    c2 = {"selection": 23.341, "heldout": 22.260}
    want = {("selection", "unclipped"): 6.82, ("heldout", "unclipped"): 6.78,
            ("selection", "clipped"): 2.20, ("heldout", "clipped"): 2.32}
    for pool, kind, ew2, mx in rows:
        k = "unclipped" if kind.startswith("unclipped") else "clipped"
        if (pool, k) not in want:
            continue
        ratio = (float(mx) - c2[pool]) / (float(ew2) - c2[pool])
        check("Eq. (5) ceiling, %s pool, %s" % (pool, k), want[(pool, k)], ratio)
else:
    print("  (audit/ceiling.txt not in this package -- the ceiling is unchecked)")

print()
print("=" * 100)
print("SECTION 5b  every factorial cell ran exactly five times")
print("=" * 100)
ledp = R / "g3_measure_vs_loss_runs.jsonl"
if ledp.exists():
    led = [json.loads(l) for l in ledp.read_text().splitlines() if l.strip()]
    counts = collections.Counter((r["student"], r["objective"], r["mode"]) for r in led)
    check("ledger records in total", 80, len(led))
    check("cells in the design", 16, len(counts))
    check("every cell carries exactly five records", True,
          set(counts.values()) == {5})
    seeds = collections.Counter((r["student"], r["objective"], r["mode"], r["seed"])
                                for r in led)
    check("no arm appears twice", True, set(seeds.values()) == {1})

    # Provenance. Sixty of the eighty records carry a code_sha; twenty predate the
    # field. The twenty are a block -- seeds 2 and 4 of EARLY, seeds 0 and 1 of
    # LATE and seed 4 of UNIFORM, in all four student-objective cells -- so the
    # matched pair draws seeds from both. Recomputing it on the stamped sixty
    # alone is the check that the split does not carry the result.
    stamped = [r for r in led if r.get("code_sha")]
    check("ledger records carrying a code_sha", 60, len(stamped))
    # the previous environment, kept so the 79.29% in Section 5 is traceable
    prev = R / "superseded" / "gp_control_prev_environment.json"
    if prev.is_file():
        pv = json.loads(prev.read_text())
        check("previous environment, teacher accuracy (%)", 79.29,
              100 * pv["teacher_accuracy"])
        check("previous environment, GPU", "NVIDIA GeForce RTX 4090",
              pv["_env"]["gpu"])
    check("they all carry the same one", 1,
          len({r["code_sha"] for r in stamped}))
    for st, nm, want in ((S05, "0.5B ce", 16.79), (S05, "0.5B fkl", 8.65),
                         (S15, "1.5B ce", 25.97), (S15, "1.5B fkl", 14.93)):
        obj = nm.split()[1]
        sel = lambda m: [100 * r["accuracy"] for r in stamped
                         if r["student"] == st and r["objective"] == obj
                         and r["mode"] == m]
        e, l = sel("early"), sel("late")
        check("  stamped-only early-late, %s (n=%d)" % (nm, len(e)), want,
              float(np.mean(e) - np.mean(l)), tol=0.002)
    check("every ledger record is in the merged blob", 80,
          sum(1 for r in g3r["runs"] if r))
else:
    print("  (the run ledger is not in this package -- cell counts are unchecked)")

print()
print("=" * 100)
print("SECTION 6b  Table 1, the five forward tests")
print("=" * 100)
fwd = R.parent / "forward_tests.json"
if fwd.exists():
    F = json.loads(fwd.read_text())
    def means(stage):
        return [a["mean_pct"] for a in F[stage]["arms"].values()]
    def c2s(stage):
        return [a["c2"] for a in F[stage]["arms"].values()]

    check("Table 1, training stages", 3, F["_totals"]["stages"])
    check("Table 1, arms", 18, F["_totals"]["arms"])
    check("Table 1, runs", 54, F["_totals"]["runs"])
    check("Table 1, no run lost to GPU memory", 0, F["_totals"]["oom_runs"])
    allc2 = c2s("ladder_g9") + c2s("positional_g10") + c2s("depth_breadth_g11")
    check("Table 1, lowest C2 over all arms", 1.10, min(allc2))
    check("Table 1, highest C2 over all arms", 30.64, max(allc2))

    # The range the text quotes is over the eleven PREDICTION arms (g10, g11).
    # Quoting it over all eighteen would be wrong, because the top of the ladder
    # falls below it; that is checked here so the two cannot be confused again.
    pred = means("positional_g10") + means("depth_breadth_g11")
    check("prediction arms, lowest accuracy (%)", 65.42, min(pred))
    check("prediction arms, highest accuracy (%)", 67.75, max(pred))
    check("the ladder's top level sits below the prediction range", True,
          min(means("ladder_g9")) < min(pred))
    if TEX:
        in_tex("the prediction-arm range, scoped to predictions",
               "every prediction arm lands between $%.2f\\%%$ and $%.2f\\%%$"
               % (min(pred), max(pred)))

    # row 1: magnitude of C2 on the ladder
    L = F["ladder_g9"]
    check("row 1, ladder C2 low", 4.87, min(c2s("ladder_g9")))
    check("row 1, ladder C2 high", 19.45, max(c2s("ladder_g9")))
    check("row 1, cliff ratio against the 3.0 bar", 7.06, L["observed"]["cliff_ratio"])
    check("row 1, ladder accuracy low (%)", 65.28, min(means("ladder_g9")))
    check("row 1, ladder accuracy high (%)", 67.72, max(means("ladder_g9")))
    check("row 1, ladder spread (points)", 2.44,
          max(means("ladder_g9")) - min(means("ladder_g9")), tol=0.01)
    check("row 1, every ladder level carries three seeds", 3,
          min(a["n_seeds"] for a in L["arms"].values()))

    # the step at the top of the ladder, which is the one bar that is cleared
    top = [a for a in L["arms"].values() if abs(a["c2"] - 19.448) < 1e-3][0]
    nxt = [a for a in L["arms"].values() if abs(a["c2"] - 16.651) < 1e-3][0]
    flat = [a["mean_pct"] for a in L["arms"].values() if a["c2"] < 19.0]
    check("row 1, six levels flat within (points)", 0.79, max(flat) - min(flat), tol=0.01)
    check("row 1, flat band low (%)", 66.94, min(flat))
    check("row 1, flat band high (%)", 67.72, max(flat))
    check("row 1, the step at the top (points)", 2.30,
          nxt["mean_pct"] - top["mean_pct"], tol=0.005)
    ta = np.array([r["accuracy_pct"] for r in nxt["runs"]])
    tb = np.array([r["accuracy_pct"] for r in top["runs"]])
    d, se, df = welch(ta, tb)
    t = t_quantile(df)
    check("row 1, step interval low", 0.77, d - t * se, tol=0.02)
    check("row 1, step interval high", 3.84, d + t * se, tol=0.02)
    drops = sorted(L["observed"]["adjacent_drops"], key=abs)
    check("row 1, mean non-maximal drop (points)", 0.33,
          float(np.mean(np.abs(drops[:-1]))), tol=0.02)
    check("row 1, verdict recorded by the stage", "threshold", L["observed"]["verdict"])
    check("row 1, threshold lower edge", 16.651, L["observed"]["threshold_between"][0])
    check("row 1, threshold upper edge", 19.448, L["observed"]["threshold_between"][1])
    check("row 1, rank correlation over the seven levels", -0.32,
          L["observed"]["rank_rho"], tol=0.01)

    # the unclipped coefficient over the same levels: the paper states its range
    # and that the ordering differs, which is why only the range is quoted
    g9b = load("g9_threshold.json")
    unc = [(v["c2"], v["c2_unclipped"]) for v in g9b["coverage"].values()
           if isinstance(v, dict) and "c2_unclipped" in v]
    check("row 1, unclipped C2 low", 206, min(u for _, u in unc), tol=0.005)
    check("row 1, unclipped C2 high", 9153, max(u for _, u in unc), tol=0.005)
    check("row 1, clipped and unclipped order the levels differently", True,
          [u for _, u in sorted(unc)] != sorted(u for _, u in unc))
    hi_unc = max(unc, key=lambda t: t[1])[0]
    hi_acc = [a["mean_pct"] for a in L["arms"].values()
              if abs(a["c2"] - hi_unc) < 1e-3][0]
    check("row 1, the largest unclipped level loses nothing", True,
          hi_acc >= max(means("ladder_g9")) - 0.01)

    # rows 2 and 3: where the starvation sits
    P = F["positional_g10"]
    check("row 2, dial span ratio", 20.8, P["observed"]["dial_span_ratio"], tol=0.005)
    check("row 2, early-series rho against the +0.80 bar", -0.60,
          P["observed"]["early_series_rho"])
    check("row 2, levels in the early series", 4,
          sum(1 for k in P["arms"] if k.startswith("early")))
    check("row 3, positional gap (points) against the 10-point bar", -0.12,
          P["observed"]["positional_gap_points"])
    early = {k: v["mean_pct"] for k, v in P["arms"].items() if k.startswith("early")}
    late = {k: v["mean_pct"] for k, v in P["arms"].items() if k.startswith("late")}
    # the registered gap is read at the tightest target, not averaged over the dial
    check("row 3, gap recomputed as late - early at the tightest target",
          P["observed"]["positional_gap_points"],
          late["late|0.02"] - early["early|0.02"], tol=0.001)
    check("rows 2-3, positional accuracy low (%)", 66.36, min(means("positional_g10")))
    check("rows 2-3, positional accuracy high (%)", 67.19, max(means("positional_g10")))

    # rows 4 and 5: depth and breadth
    B = F["depth_breadth_g11"]
    check("rows 4-5, reference accuracy (%)", 66.6, B["reference_accuracy_pct"])
    check("row 4, depth cell change (points)", -1.18,
          B["observed"]["b1_deep"]["change_vs_reference_pts"])
    check("row 5, two-bucket cell change (points)", 1.15,
          B["observed"]["b1b2"]["change_vs_reference_pts"])
    check("row 5, four-bucket cell change (points)", -0.73,
          B["observed"]["b1b2b3b4"]["change_vs_reference_pts"])
    for k in ("b1_deep", "b1b2", "b1b2b3b4"):
        check("rows 4-5, %s change recomputed from the arm mean" % k,
              B["observed"][k]["change_vs_reference_pts"],
              B["arms"][k]["mean_pct"] - B["reference_accuracy_pct"], tol=0.01)
    check("rows 4-5, depth/breadth accuracy low (%)", 65.42, min(means("depth_breadth_g11")))
    check("rows 4-5, depth/breadth accuracy high (%)", 67.75, max(means("depth_breadth_g11")))
    check("row 5, four-bucket cell C2", 15.57, B["arms"]["b1b2b3b4"]["c2"])

    # the three comparisons the text rests on
    check("ruling-out comparison, ladder level at C2=16.65 costs nothing",
          True, max(a["mean_pct"] for a in L["arms"].values()
                    if abs(a["c2"] - 16.65) < 0.01) > L["base_accuracy_pct"] - 1.0)

    # every arm sits far above the catastrophic LATE arm of the factorial
    check("all forward-test arms stay above 64% (no cliff anywhere)", True,
          min(means("ladder_g9") + means("positional_g10")
              + means("depth_breadth_g11")) > 64.0)

    # the bars in Table 1 are the ones the pre-registration file holds
    thr = R.parent / "code" / "thresholds.json"
    if thr.exists():
        T = json.loads(thr.read_text())
        check("registered bar, cliff ratio", 3.0, T["g9"]["cliff_ratio_min"])
        check("registered bar, early-series rho", 0.80, T["g10"]["early_series_rho_min"])
        check("registered bar, positional gap (points)", 10.0,
              T["g10"]["positional_gap_min_points"])
        check("registered bar, depth and breadth drop (points)", 10.0,
              T["g11"]["collapse_drop_min_points"])
        check("registered floor for the depth/breadth stage (steps)",
              B["matched_steps"], T["g11"]["floor_steps"])
        # each outcome falls short of its own bar, which is what "null" means here
        # the one predictor that clears its bar; the text reports the step it resolves
        check("outcome 1 clears its bar", True,
              L["observed"]["cliff_ratio"] >= T["g9"]["cliff_ratio_min"])
        check("outcome 2 below its bar", True,
              P["observed"]["early_series_rho"] < T["g10"]["early_series_rho_min"])
        check("outcome 3 below its bar", True,
              P["observed"]["positional_gap_points"]
              < T["g10"]["positional_gap_min_points"])
        for k in ("b1_deep", "b1b2", "b1b2b3b4"):
            check("outcome for %s below its bar" % k, True,
                  -B["observed"][k]["change_vs_reference_pts"]
                  < T["g11"]["collapse_drop_min_points"])
    else:
        print("  (thresholds.json absent -- the registered bars are unchecked)")
else:
    print("  (forward_tests.json absent -- Table 1 is unchecked)")

print()
print("=" * 100)
print("SECTION 8  the manuscript's printed numbers, recomputed and matched to conf.tex")
print("=" * 100)
if TEXT is None:
    print("  (conf.tex is not in this package)")
else:
    def ci(a, b_, nd=2):
        """Mean difference and its Welch interval, rendered as the paper prints them."""
        d, se, df = welch(a, b_)
        t = t_quantile(df)
        return d, d - t * se, d + t * se

    # the abstract's four early-minus-late values and the widest of their intervals
    gaps = []
    for st, obj in ((S05, "ce"), (S05, "fkl"), (S15, "ce"), (S15, "fkl")):
        gaps.append(ci(cell(st, obj, "early"), cell(st, obj, "late")))
    ordered = sorted((g[0] for g in gaps), reverse=True)
    in_tex("abstract, the four early-minus-late values",
           "by $%.1f$, $%.1f$, $%.1f$ and $%.1f$ points"
           % tuple(ordered))
    top = max(gaps, key=lambda g: g[0])
    in_tex("abstract, the interval on the largest",
           "Welch interval of $[%.1f,%.1f]$" % (top[1], top[2]))

    # EARLY against UNIFORM: the paper must say that EARLY is the lower of the
    # two, because the matched pair is a cost of dropping the opening steps and
    # not a gain from concentrating supervision on them
    eu = []
    for st, obj in ((S15, "ce"), (S15, "fkl"), (S05, "ce"), (S05, "fkl")):
        eu.append(ci(cell(st, obj, "uniform"), cell(st, obj, "early")))
    check("early is below uniform in all four cells", 4, sum(d > 0 for d, _, _ in eu))
    check("three of the four early-uniform intervals exclude zero", 3,
          sum(lo * hi > 0 for _, lo, hi in eu))
    in_tex("early does not improve on uniform",
           "lower in all four cells, by $%.2f$, $%.2f$, $%.2f$ and $%.2f$ points"
           % tuple(sorted(d for d, _, _ in eu)))

    # the widest seed spread of LATE, which the text rounds to a whole number
    spread = max(cell(st, obj, "late").max() - cell(st, obj, "late").min()
                 for st in (S05, S15) for obj in ("ce", "fkl"))
    in_tex("the widest LATE seed spread",
           "spread by up to $%d$ points" % round(spread))

    # the positive control's single-arm detectable difference
    in_tex("the positive control's single-arm MDE",
           "positive control's $%.2f$ from the single-arm form" % gpc["seed_mde_points"])

    # the three deltas of the quantity sweep
    d3 = []
    for pr in (100, 1000, 6000):
        lo = np.mean([100 * x["accuracy"] for x in by[(pr, 25000)]])
        hi = np.mean([100 * x["accuracy"] for x in by[(pr, 100000)]])
        d3.append(hi - lo)
    in_tex("the three deltas of the quantity sweep",
           "moves accuracy by $%+.2f$, $%+.2f$ and $%+.2f$ points" % tuple(d3))

    # soft targets under cluster-drop, with Welch degrees of freedom
    for st, nm in ((S05, "0.5B"), (S15, "1.5B")):
        d, lo, hi = ci(cell(st, "fkl", "cluster_drop"), cell(st, "ce", "cluster_drop"))
        in_tex("soft targets under cluster-drop, %s" % nm,
               "$%+.2f$ $[%.2f,%.2f]$" % (d, lo, hi))

    # the variance ratios quoted for the matched pair, each beside the cell it
    # belongs to: the paper names them because the four are not in cell order
    for st, obj, nm in ((S05, "fkl", "0.5B \\textsc{fkl}"),
                        (S05, "ce", "0.5B \\textsc{ce}"),
                        (S15, "ce", "1.5B \\textsc{ce}"),
                        (S15, "fkl", "1.5B \\textsc{fkl}")):
        r = cell(st, obj, "late").var(ddof=1) / cell(st, obj, "early").var(ddof=1)
        in_tex("variance ratio, %s" % nm.replace("\\textsc{", "").replace("}", ""),
               "$%d$ (%s)" % (round(r), nm))

    # the ladder: the flat band, the step and its interval
    if TEX and fwd.exists():
        arms_ = F["ladder_g9"]["arms"]
        topa = [a for a in arms_.values() if abs(a["c2"] - 19.448) < 1e-3][0]
        nxta = [a for a in arms_.values() if abs(a["c2"] - 16.651) < 1e-3][0]
        flat_ = [a["mean_pct"] for a in arms_.values() if a["c2"] < 19.0]
        in_tex("the ladder's flat band",
               "within $%.2f$ points of one another, between $%.2f\\%%$ and $%.2f\\%%$"
               % (max(flat_) - min(flat_), min(flat_), max(flat_)))
        ta = np.array([r["accuracy_pct"] for r in nxta["runs"]])
        tb = np.array([r["accuracy_pct"] for r in topa["runs"]])
        d, lo, hi = ci(ta, tb)
        in_tex("the ladder's step and interval",
               "a step of $%.2f$ points with a Welch interval of $[%.2f,%.2f]$" % (d, lo, hi))
        in_tex("the ladder's cliff ratio",
               "is $%.2f$ against" % F["ladder_g9"]["observed"]["cliff_ratio"])

print()
print("=" * 100)
print("%d checks OK, %d MISMATCH" % (ok_count, bad_count))
print("=" * 100)
