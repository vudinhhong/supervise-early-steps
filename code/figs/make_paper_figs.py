#!/usr/bin/env python
"""Draw the two figures the manuscript uses, from results/ only.

    python make_paper_figs.py [outdir]

Nothing is hard-coded: every value is read from the result blobs, so a figure
cannot drift away from the numbers the text states. Output is vector PDF sized
for the LNCS text width (122 mm).
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent.parent / "results"
OUT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent.parent.parent

S05, S15 = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"
MASKS = ["uniform", "early", "late", "cluster_drop"]
MLAB = ["uniform", "early", "late", "cluster-\ndrop"]
TW = 4.80          # text width in inches (122 mm)

# Okabe-Ito, which survives the common forms of colour blindness. Colour carries
# the objective and the dash pattern carries the student, so the figures still
# read in a greyscale print run.
BLUE, VERM, SKY, ORANGE = "#0072B2", "#D55E00", "#56B4E9", "#E69F00"
GREEN = "#009E73"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 6.8,
    "axes.labelsize": 6.8,
    "axes.titlesize": 7.6,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "lines.linewidth": 0.9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def load(name):
    return json.loads((R / name).read_text())


g3 = load("g3_measure_vs_loss.json")
g9 = load("g9_threshold.json")


def cell(st, obj, mode):
    return np.array([100 * r["accuracy"] for r in g3["runs"]
                     if r["student"] == st and r["objective"] == obj
                     and r["mode"] == mode])


def t_quantile(df, q=0.975):
    xs = np.linspace(0.0, 60.0, 600001)
    pdf = (1.0 + xs * xs / df) ** (-(df + 1.0) / 2.0)
    cdf = np.concatenate(([0.0], np.cumsum((pdf[1:] + pdf[:-1]) * 0.5 * (xs[1] - xs[0]))))
    return float(np.interp(q - 0.5, cdf / (2.0 * cdf[-1]), xs))


def ci(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = float(a.mean() - b.mean())
    t = t_quantile(df)
    return d, d - t * se, d + t * se


# ---------------------------------------------------------------- figure 1
# Every seed of every cell on one axis. The table gives means and standard
# deviations; what it cannot show is that the spread belongs to two particular
# masks, and that LATE is bimodal rather than merely noisy.
def fig_cells(path):
    series = [(S15, "fkl", "1.5B, FKL", "o", "-", BLUE),
              (S15, "ce", "1.5B, CE", "s", "-", VERM),
              (S05, "fkl", "0.5B, FKL", "^", "--", SKY),
              (S05, "ce", "0.5B, CE", "v", "--", ORANGE)]
    fig, ax = plt.subplots(figsize=(TW, 1.56))
    xs = np.arange(4)
    for k, (st, obj, lab, mk, ls, col) in enumerate(series):
        means = np.array([cell(st, obj, m).mean() for m in MASKS])
        # the four series share each x position, so their seed clouds are pushed
        # apart before the within-series jitter is added
        off = (k - 1.5) * 0.085
        for i, m in enumerate(MASKS):
            v = cell(st, obj, m)
            jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.022
            ax.scatter(xs[i] + off + jit, v, s=3.6, facecolors="none",
                       edgecolors=col, linewidths=0.5, alpha=0.75, zorder=2)
        ax.plot(xs, means, ls, color=col, lw=1.0, marker=mk, ms=3.4,
                mfc="white", mew=1.0, label=lab, zorder=4)
    # the two untrained accuracies: 0.5B starts above 1.5B, so each dotted line
    # is labelled rather than sharing one caption word.
    for st, txt, xt, dy, va in ((S05, "0.5B untrained", 0.95, 1.4, "bottom"),
                                (S15, "1.5B untrained", 1.95, -1.4, "top")):
        y = 100 * g3["base"][st]["accuracy"]
        ax.axhline(y, color="0.6", lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.text(xt, y + dy, txt, fontsize=6.0, color="0.45", ha="left", va=va)
    ax.set_xticks(xs)
    ax.set_xticklabels(["uniform\n$C_2{=}1.0$", "early\n$C_2{=}5.4$",
                        "late\n$C_2{=}14.5$", "cluster-drop\n$C_2{=}39.1$"])
    ax.set_xlim(-0.35, 3.55)
    ax.set_ylim(-5.5, 74)   # room under the 1.5B line for its label
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("accuracy (%)")
    ax.tick_params(axis="x", pad=2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.20), frameon=False,
              handlelength=2.0, borderaxespad=0.0, labelspacing=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 2
# The two contrasts the paper rests on as intervals rather than as a list of
# numbers in running text, and beside them the coverage ladder, whose seven
# levels are the evidence that the coefficient does not order accuracy.
def fig_contrasts(path):
    fig, (axl, axr, axc) = plt.subplots(
        1, 3, figsize=(TW, 1.40), gridspec_kw={"width_ratios": [1.0, 1.0, 0.92]})

    col = {("ce", S15): VERM, ("fkl", S15): BLUE,
           ("ce", S05): ORANGE, ("fkl", S05): SKY}
    rows = [(S15, "ce", "1.5B, CE"), (S15, "fkl", "1.5B, FKL"),
            (S05, "ce", "0.5B, CE"), (S05, "fkl", "0.5B, FKL")]
    for k, (st, obj, lab) in enumerate(rows):
        d, lo, hi = ci(cell(st, obj, "early"), cell(st, obj, "late"))
        c = col[(obj, st)]
        axl.plot([lo, hi], [k, k], color=c, lw=1.0, solid_capstyle="butt")
        axl.plot([lo, lo], [k - 0.13, k + 0.13], color=c, lw=1.0)
        axl.plot([hi, hi], [k - 0.13, k + 0.13], color=c, lw=1.0)
        axl.scatter([d], [k], s=11, color=c, zorder=4)
    axl.axvline(0, color="0.55", lw=0.7, ls=(0, (3, 2)))
    axl.set_yticks(range(len(rows)))
    axl.set_yticklabels([r[2] for r in rows])
    axl.set_ylim(-0.6, len(rows) - 0.4)
    axl.set_xlim(-2.5, 36.5)
    axl.set_xlabel("early $-$ late (points)")
    axl.set_title("(a) the matched pair", pad=3)

    rows2 = []
    for st, nm in ((S15, "1.5B"), (S05, "0.5B")):
        for m, ml in zip(MASKS, ["uniform", "early", "late", "cluster-drop"]):
            rows2.append((st, m, "%s, %s" % (nm, ml)))
    for k, (st, m, lab) in enumerate(rows2):
        d, lo, hi = ci(cell(st, "fkl", m), cell(st, "ce", m))
        # red where destroying coverage makes the objective matter, blue elsewhere
        c = VERM if m == "cluster_drop" else BLUE
        axr.plot([lo, hi], [k, k], color=c, lw=1.0, solid_capstyle="butt")
        axr.plot([lo, lo], [k - 0.13, k + 0.13], color=c, lw=1.0)
        axr.plot([hi, hi], [k - 0.13, k + 0.13], color=c, lw=1.0)
        axr.scatter([d], [k], s=11, color=c, zorder=4)
    axr.axvline(0, color="0.55", lw=0.7, ls=(0, (3, 2)))
    axr.set_yticks(range(len(rows2)))
    axr.set_yticklabels([r[2] for r in rows2])
    axr.set_ylim(-0.6, len(rows2) - 0.4)
    axr.set_xlabel("FKL $-$ CE (points)")
    axr.set_title("(b) the objective, by mask", pad=3)

    acc = {}
    for r in g9["runs"]:
        acc.setdefault(round(r["c2"], 3), []).append(100 * r["accuracy"])
    levels = sorted(acc)
    mean = np.array([np.mean(acc[c]) for c in levels])
    sd = np.array([np.std(acc[c], ddof=1) for c in levels])
    flat = mean[:-1]
    axc.axhspan(flat.min(), flat.max(), color=GREEN, alpha=0.15, zorder=0)
    axc.errorbar(levels[:-1], mean[:-1], yerr=sd[:-1], fmt="o", ms=2.8,
                 color=GREEN, ecolor=GREEN, elinewidth=0.7, capsize=1.4,
                 capthick=0.7, zorder=3)
    axc.errorbar(levels[-1:], mean[-1:], yerr=sd[-1:], fmt="o", ms=2.8,
                 color=VERM, ecolor=VERM, elinewidth=0.7, capsize=1.4,
                 capthick=0.7, zorder=3)
    axc.plot(levels, mean, color="0.55", lw=0.7, zorder=2)
    xa = levels[-1] + 0.9
    axc.annotate("", xy=(xa, mean[-1]), xytext=(xa, mean[-2]),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color="black",
                                 shrinkA=0.5, shrinkB=0.5))
    axc.text(xa - 0.6, (mean[-1] + mean[-2]) / 2, "%.2f" % (mean[-2] - mean[-1]),
             ha="right", va="center", fontsize=6.0)
    axc.set_xlabel("$C_2$ (clipped)")
    axc.set_ylabel("accuracy (%)", labelpad=1.5)
    axc.set_xlim(3.4, 22.6)
    axc.set_ylim(64.8, 68.5)
    axc.set_yticks([65, 66, 67, 68])
    axc.set_title("(c) the coverage ladder", pad=3)
    axc.tick_params(axis="y", pad=1.5)

    for ax in (axl, axr, axc):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in (axl, axr):
        ax.tick_params(axis="y", length=0, pad=1.5)
    fig.subplots_adjust(wspace=0.60)
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_cells(OUT / "fig_cells.pdf"),
              fig_contrasts(OUT / "fig_contrasts.pdf")):
        print("wrote %s (%.1f kB)" % (f.name, f.stat().st_size / 1024))
