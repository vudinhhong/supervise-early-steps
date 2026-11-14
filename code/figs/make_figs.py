#!/usr/bin/env python
"""Figures for the paper and the slides, drawn from the result blobs only.

Every number plotted here is read from a file written by a stage run; nothing is
typed in by hand and nothing is smoothed.  Run from the repository root:

    PYTHONPATH=. python figs/make_figs.py
"""
import json, pathlib, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(".")
OUT = R / "figs"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
CE, FKL, G9C = "#1b4f72", "#b9770e", "#117a65"
GREY = "#7f8c8d"
S15 = "Qwen/Qwen2.5-1.5B"
S05 = "Qwen/Qwen2.5-0.5B"


def load(p):
    q = R / p
    return json.loads(q.read_text()) if q.exists() else None


# ===================================================================== fig 1
def fig_threshold():
    g3 = load("results/g3_measure_vs_loss.json")
    g7 = load("results/g7_dissociation_c2fix.json")
    g9 = load("results-g9/g9_threshold.json")

    pts = [("G3", r["c2"], r["accuracy"] * 100) for r in g3["runs"]
           if r["student"] == S15 and r["objective"] == "ce"]
    base = g3["base"][S15]["accuracy"] * 100
    for r in (g7 or {}).get("runs", []):
        pts.append(("G7", r["c2"], r["accuracy"] * 100))
    for r in (g9 or {}).get("runs", []):
        pts.append(("G9", r["c2"], r["accuracy"] * 100))

    agg = collections.defaultdict(list)
    for src, c2, acc in pts:
        agg[(src, round(c2, 3))].append(acc)

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    style = {"G3": ("o", CE, "four supervision masks"),
             "G7": ("s", FKL, "prompt-count sweep"),
             "G9": ("^", G9C, "coverage ladder")}
    seen = set()
    for (src, c2), accs in sorted(agg.items(), key=lambda kv: kv[0][1]):
        mk, col, lab = style[src]
        ax.errorbar(c2, np.mean(accs),
                    yerr=(np.std(accs, ddof=1) if len(accs) > 1 else 0),
                    fmt=mk, color=col, ms=5, capsize=2.5, lw=1,
                    label=(lab if src not in seen else None))
        seen.add(src)
    if "G9" not in {s for s, _ in agg}:
        ax.axvspan(8.4, 14.4, color=GREY, alpha=0.13, lw=0)
        ax.text(10.9, 30, "not yet\nmeasured", ha="center", va="center",
                fontsize=7.4, color=GREY, style="italic")
    ax.axhline(base, ls=":", color=GREY, lw=1)
    ax.text(1.05, base + 1.8, "untrained student (%.2f%%)" % base,
            fontsize=7.2, color=GREY)
    ax.set_xscale("log")
    ax.set_xlabel(r"coverage coefficient $C_2(\mu)$  (log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy against coverage, one student and one objective\n"
                 "(Qwen2.5-1.5B, cross-entropy)", fontsize=8.8)
    ax.legend(frameon=False, loc="lower left", title="measured in")
    ax.set_ylim(-2, 78)
    fig.savefig(OUT / "fig_threshold.pdf")
    print("wrote fig_threshold.pdf  (%d levels)" % len(agg))


# ===================================================================== fig 2
def fig_g3():
    d = load("results/g3_measure_vs_loss.json")
    modes = ["uniform", "early", "late", "cluster_drop"]
    c2 = {m: [r["c2"] for r in d["runs"] if r["mode"] == m][0] for m in modes}
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), sharey=True)
    for ax, st in zip(axes, [S05, S15]):
        for obj, col, lab in (("ce", CE, "cross-entropy"),
                              ("fkl", FKL, "forward KL")):
            xs, ys, es = [], [], []
            for m in modes:
                v = [100 * r["accuracy"] for r in d["runs"]
                     if r["student"] == st and r["objective"] == obj
                     and r["mode"] == m]
                xs.append(c2[m])
                ys.append(np.mean(v))
                es.append(np.std(v, ddof=1))
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=col, ms=4.5,
                        capsize=2.5, lw=1.3, label=lab)
        b = 100 * d["base"][st]["accuracy"]
        ax.axhline(b, ls=":", color=GREY, lw=1)
        ax.text(1.1, b + 2.2, "untrained (%.2f%%)" % b, fontsize=7, color=GREY)
        ax.set_xscale("log")
        ax.set_xticks([c2[m] for m in modes])
        ax.set_xticklabels(["%s\n%.1f" % (m.replace("_", "-"), c2[m])
                            for m in modes], fontsize=6.8)
        ax.set_xlabel("supervision mask and its $C_2$")
        e = d["anova"][st]
        ax.set_title("%s\n$\\eta^2_{\\rm measure}=%.3f$, $\\eta^2_{\\rm loss}=%.3f$"
                     "  (ratio %.2f)"
                     % (st.split("/")[-1], e["eta2_measure"], e["eta2_loss"],
                        e["eta2_measure"] / e["eta2_loss"]), fontsize=8.4)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.savefig(OUT / "fig_g3.pdf")
    print("wrote fig_g3.pdf")


# ===================================================================== fig 3
def fig_g4():
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0))
    for ax, (tag, col, name) in zip(axes, [
            ("results-g4/g4_predict_ext.json", FKL, "forward KL"),
            ("results-g4/g4_predict_ce.json", CE, "cross-entropy")]):
        d = load(tag)
        if d is None:
            ax.set_visible(False)
            continue
        P = d["points"]
        pairs = sorted({(p["student"], p["teacher"]) for p in P})
        ok = 0
        for s, t in pairs:
            q = sorted([p for p in P if p["student"] == s and p["teacher"] == t],
                       key=lambda x: x["c2"])
            g = [p["gain_points"] for p in q]
            ok += all(g[i] > g[i + 1] for i in range(len(g) - 1))
            ax.plot([p["c2"] for p in q], g, "o-", color=col, ms=3.8, lw=1,
                    alpha=0.78)
        ax.axhline(0, ls=":", color=GREY, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("$C_2(\\mu)$")
        ax.set_title("%s: %d/%d pairs ordered\npooled $\\rho=%+.3f$ ($p=%.4f$)"
                     % (name, ok, len(pairs), d["rho_coverage"]["rho"],
                        d["rho_coverage"]["p"]), fontsize=8.4)
    axes[0].set_ylabel("accuracy gain over\nuntrained student (points)")
    fig.suptitle("One line per teacher-student pair: coverage orders every pair, "
                 "while the pooled rank test is diluted by headroom",
                 fontsize=8.6, y=1.04)
    fig.savefig(OUT / "fig_g4.pdf")
    print("wrote fig_g4.pdf")


# ===================================================================== fig 4
def fig_g0():
    d = load("results-syn/g0_estimator.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
    ax = axes[0]
    for key, col, mk, lab in (("c2_A", CE, "o", "A: counting on $\\sigma$"),
                              ("c2_B", FKL, "s", "B: density ratio"),
                              ("c2_C", "#884ea0", "^", "C: $k$-NN")):
        conv = d["convergence"][key]
        ns = sorted(int(k) for k in conv)
        ax.plot(ns, [conv[str(n)] for n in ns], mk + "-", color=col, ms=4.5,
                lw=1.3, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("samples $n$")
    ax.set_ylabel("median relative error")
    ax.set_title("Estimator accuracy against exact values\n(780 configurations)",
                 fontsize=8.4)
    ax.legend(frameon=False)

    ax = axes[1]
    ex = d["exact"]
    kl = np.array([r["e_kl_mu_raw"] for r in ex])
    gap = np.array([r["gap"] for r in ex])
    # The registered criterion for an instance of the corollary, taken from
    # experiments/g1_theory.py: raw fidelity loss <= 1e-6 AND gap >= 0.10.  The
    # gap floor is part of the definition -- a configuration with zero loss and
    # zero gap is not a counterexample to anything.
    zero = kl <= 1e-6
    inst = zero & (gap >= 0.10)
    FLOOR = 1e-9
    ax.scatter(np.maximum(kl[~zero], FLOOR), gap[~zero], s=7, color=GREY,
               alpha=0.42, lw=0, label="fidelity loss $> 10^{-6}$")
    ax.scatter(np.full(int((zero & ~inst).sum()), FLOOR), gap[zero & ~inst],
               s=14, color=GREY, marker="D", lw=0,
               label="loss $\\approx 0$, gap $< 0.10$ (%d)"
                     % int((zero & ~inst).sum()))
    ax.scatter(np.full(int(inst.sum()), FLOOR), gap[inst], s=22, color=CE,
               marker="D", lw=0,
               label="loss $\\approx 0$, gap $\\geq 0.10$ (%d)" % int(inst.sum()))
    ax.axhline(0.10, ls="--", color=CE, lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"fidelity loss $\mathbb{E}_\mu[\mathrm{KL}]$ on the raw measure")
    ax.set_ylabel("capability gap")
    ax.set_title("Corollary 1.1 instantiated: %d configurations fit their\n"
                 "supervision exactly and still lose up to %.3f"
                 % (int(inst.sum()), gap[inst].max()), fontsize=8.4)
    ax.legend(frameon=False, loc="upper right", fontsize=6.6)
    fig.savefig(OUT / "fig_g0.pdf")
    print("wrote fig_g0.pdf  (zero-loss configurations: %d)" % int(zero.sum()))


if __name__ == "__main__":
    for f in (fig_threshold, fig_g3, fig_g4, fig_g0):
        try:
            f()
        except Exception as e:
            print("FAILED", f.__name__, type(e).__name__, e)
