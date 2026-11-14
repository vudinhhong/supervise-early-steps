r"""The statistics the protocol needs, and nothing else.

Four of these exist because the previous round's conclusions were not supported
by its designs.

``mde``                 what difference the design can detect at all.
``paired_bootstrap``    a confidence interval for a difference of accuracies on
                        the same items.
``reliability``         how much of the between-configuration variance in a
                        dependent variable is signal.  A correlation study whose
                        dependent variable has reliability 0.24 cannot observe a
                        correlation above 0.46 however true the hypothesis is.
``disattenuate``        the correction, so "rho was 0.5" becomes "rho was 0.5,
                        the design's ceiling was 0.70, corrected 0.71".
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


def mde(n: int, q: float = 0.20, power_z: float = 2.8) -> float:
    """Minimum detectable paired difference in accuracy, percentage points."""
    return 100.0 * power_z * (q / max(n, 1)) ** 0.5


def paired_bootstrap(a: Sequence[int], b: Sequence[int], n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """Bootstrap over items for ``mean(a) - mean(b)`` on the same items."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    assert len(a) == len(b), "paired bootstrap needs aligned per-item outcomes"
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    draws = d[idx].mean(1)
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.quantile(draws, 0.025)),
            "hi": float(np.quantile(draws, 0.975)), "p": float(min(1.0, p)),
            "n_items": int(len(d)), "disagree_rate": float((d != 0).mean())}


def seed_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                   seed: int = 0) -> dict:
    """Difference of two arms' per-seed accuracies, bootstrapped over seeds."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    draws = da - db
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"diff": float(a.mean() - b.mean()),
            "lo": float(np.quantile(draws, 0.025)),
            "hi": float(np.quantile(draws, 0.975)),
            "p": float(min(1.0, p)), "n_seeds": [len(a), len(b)],
            "sd": [float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
                   float(b.std(ddof=1)) if len(b) > 1 else float("nan")]}


def reliability(per_config_seed_values: Sequence[Sequence[float]]) -> dict:
    r"""Variance components and the reliability of a per-configuration mean.

    ``rel = sigma_b^2 / (sigma_b^2 + sigma_w^2 / S)``.  ``sigma_w`` is pooled
    within-configuration (seed) spread; ``sigma_b`` is the between-configuration
    spread of the *true* means, estimated by subtracting the sampling part from
    the observed spread of the means and flooring at zero.
    """
    rows = [np.asarray(v, float) for v in per_config_seed_values if len(v) >= 2]
    if len(rows) < 3:
        return {"reliability": float("nan"), "note": "need >=3 configs with >=2 seeds"}
    S = float(np.mean([len(r) for r in rows]))
    sw2 = float(np.mean([r.var(ddof=1) for r in rows]))
    means = np.array([r.mean() for r in rows])
    sb2 = max(means.var(ddof=1) - sw2 / S, 0.0)
    rel = sb2 / (sb2 + sw2 / S) if (sb2 + sw2 / S) > 0 else float("nan")
    return {"reliability": float(rel), "sigma_w": float(math.sqrt(sw2)),
            "sigma_b": float(math.sqrt(sb2)), "S": S, "n_configs": len(rows)}


def disattenuate(rho_obs: float, rel_x: float, rel_y: float) -> dict:
    ceiling = math.sqrt(max(rel_x, 0.0) * max(rel_y, 0.0))
    corr = rho_obs / ceiling if ceiling > 1e-9 else float("nan")
    return {"rho_obs": float(rho_obs), "design_ceiling": float(ceiling),
            "rho_corrected": float(max(-1.0, min(1.0, corr)))}


def spearman(x: Sequence[float], y: Sequence[float]) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    n = int(ok.sum())
    r = sps.spearmanr(x[ok], y[ok])
    rho, p, how = float(r.statistic), float(r.pvalue), "asymptotic"

    # At small n the asymptotic p-value is useless: it rests on a t statistic,
    # and at |rho| = 1 that statistic is infinite, so the p-value comes back as
    # exactly zero.  With n = 4 the smallest two-sided p-value that can occur is
    # 2/4! = 0.083.  Reporting zero would be a false claim about significance,
    # so the exact value is computed by enumerating the permutations.
    xs, ys = x[ok], y[ok]
    if n <= 8 and len(set(xs)) == n and len(set(ys)) == n:
        import itertools

        def _rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            out = [0] * len(v)
            for pos, i in enumerate(order):
                out[i] = pos + 1
            return out

        rx, ry = _rank(list(xs)), _rank(list(ys))
        denom = n * (n * n - 1)
        obs = abs(1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / denom)
        hit = tot = 0
        for perm in itertools.permutations(ry):
            rr = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, perm)) / denom
            tot += 1
            if abs(rr) >= obs - 1e-12:
                hit += 1
        p, how = hit / tot, "exact (%d! permutations)" % n
    return {"rho": rho, "p": p, "n": n, "p_method": how}


def pearson(x, y) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return {"r": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    r = sps.pearsonr(x[ok], y[ok])
    return {"r": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


def williams_test(r_xy: float, r_zy: float, r_xz: float, n: int) -> dict:
    """Hotelling--Williams test that two *dependent* correlations differ.

    Needed for gate G4.3: ``corr(product, gain)`` and ``corr(fidelity, gain)``
    share the variable ``gain``, so comparing them with two independent tests
    overstates the evidence.
    """
    if n < 6 or not all(map(np.isfinite, (r_xy, r_zy, r_xz))):
        return {"t": float("nan"), "p": float("nan"), "n": n}
    R = (1 - r_xy ** 2 - r_zy ** 2 - r_xz ** 2 + 2 * r_xy * r_zy * r_xz)
    denom = 2 * (n - 1) / (n - 3) * R + ((r_xy + r_zy) ** 2 / 4) * (1 - r_xz) ** 3
    if denom <= 0:
        return {"t": float("nan"), "p": float("nan"), "n": n}
    t = (r_xy - r_zy) * math.sqrt((n - 1) * (1 + r_xz) / denom)
    p = 2 * (1 - sps.t.cdf(abs(t), df=n - 3))
    return {"t": float(t), "p": float(p), "n": n}


def partial_regression(y, X: Dict[str, Sequence[float]]) -> dict:
    """OLS with t-statistics; the design matrix gets an intercept."""
    names = list(X)
    M = np.column_stack([np.asarray(X[k], float) for k in names])
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(M).all(1)
    M, y = M[ok], y[ok]
    A = np.column_stack([np.ones(len(y)), M])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = max(len(y) - A.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tstat = beta / np.where(se > 0, se, np.nan)
    pval = 2 * (1 - sps.t.cdf(np.abs(tstat), df=dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"names": ["intercept"] + names,
            "beta": [float(b) for b in beta], "se": [float(s) for s in se],
            "t": [float(t) for t in tstat], "p": [float(p) for p in pval],
            "r2": float(1 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(len(y))}


def eta_squared(values: Dict[Tuple[str, str], Sequence[float]]) -> dict:
    r"""Two-way ANOVA effect sizes for gate G3.

    Keys are ``(loss, measure)``; values the replicate accuracies.  Reports
    ``eta^2`` for each main effect and the interaction -- the quantity G3.1
    thresholds, because "which factor explains more variance" is the actual
    question and a p-value does not answer it.
    """
    keys = list(values)
    a_levels = sorted({k[0] for k in keys})
    b_levels = sorted({k[1] for k in keys})
    allv = np.concatenate([np.asarray(values[k], float) for k in keys])
    gm = allv.mean()
    ss_tot = float(((allv - gm) ** 2).sum())

    def ss_main(which: int, levels):
        s = 0.0
        for lv in levels:
            v = np.concatenate([np.asarray(values[k], float)
                                for k in keys if k[which] == lv])
            s += len(v) * (v.mean() - gm) ** 2
        return float(s)

    ss_a, ss_b = ss_main(0, a_levels), ss_main(1, b_levels)
    ss_cells = float(sum(len(values[k]) * (np.asarray(values[k], float).mean() - gm) ** 2
                         for k in keys))
    ss_ab = max(ss_cells - ss_a - ss_b, 0.0)
    return {"eta2_loss": ss_a / ss_tot if ss_tot else float("nan"),
            "eta2_measure": ss_b / ss_tot if ss_tot else float("nan"),
            "eta2_interaction": ss_ab / ss_tot if ss_tot else float("nan"),
            "ss_total": ss_tot, "levels_loss": a_levels, "levels_measure": b_levels}


def holm(pvalues: Dict[str, float], alpha: float = 0.05) -> dict:
    """Holm--Bonferroni over the pre-registered family of key comparisons."""
    items = sorted([(v, k) for k, v in pvalues.items() if np.isfinite(v)])
    m, out, ok = len(items), {}, True
    for i, (p, k) in enumerate(items):
        thr = alpha / (m - i)
        ok = ok and (p <= thr)
        out[k] = {"p": float(p), "threshold": float(thr), "reject": bool(ok)}
    return out
