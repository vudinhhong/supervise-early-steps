r"""Three estimators of the coverage coefficient

    C2(mu) = E_{h ~ d}[ d(h) / mu(h) ] = sum_h d(h)^2 / mu(h) = 1 + chi2(d || mu)

where ``d`` is the teacher's state-occupancy measure and ``mu`` the state measure
of the distillation supervision.

Why three.  The evidential-lift study had one estimator of its central quantity
and discovered only at the end that a second, equally legitimate one correlated
with it at Spearman -0.035.  A quantity with a single estimator has not been
measured, it has been defined.  These three have disjoint failure modes:

  A  counting on the arithmetic abstraction sigma        coarse, unbiased-in-rank,
                                                         paraphrase-invariant
  B  density ratio from a logistic classifier on the     fine, but can answer with
     student's hidden state                              style instead of state
  C  k-nearest-neighbour two-sample statistic on the     no parametric model, but
     same features                                       poor in high dimension

Gate G0 checks A, B, C against an exactly computable C2; gate G2 checks that B
is not reading style; the identification gate requires they agree in rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

CLIP_M = 50.0          # ratio clip; C2 estimates are capped at this by construction


# --------------------------------------------------------------- estimator A


def histogram(codes: Sequence[int], n_states: int) -> np.ndarray:
    h = np.bincount(np.asarray(codes, dtype=np.int64), minlength=n_states)
    return h.astype(np.float64)


def c2_counting(d_codes: Sequence[int], mu_codes: Sequence[int],
                n_states: int, alpha: float = 0.5,
                clip: float = CLIP_M, total_alpha: float | None = None) -> dict:
    r"""C2 by counting on a discrete state space.

    Two guards.  Additive smoothing ``alpha`` keeps the ratio finite where
    ``mu`` has no mass and ``d`` does -- without it a single unvisited state
    sends C2 to infinity and the number stops being comparable across
    configurations.  The clip then bounds the influence of the remaining thin
    states, trading a known downward bias for a variance that does not depend
    on the rarest cell.  Both the clipped and the unclipped value are returned
    so the trade is visible rather than hidden in a default.

    ``total_alpha`` exists because a per-state ``alpha`` makes values from
    different state-space sizes incomparable, and one gate was comparing exactly
    those.  Coarsening the abstraction from 1296 cells to 54 cuts the total
    pseudo-mass ``alpha * n_states`` by a factor of 24, which sharpens every
    ratio for reasons that have nothing to do with the abstraction.  Holding the
    *total* fixed removes that confound.

    How much it matters depends on how dense the counts are, and the defensible
    numbers are worth recording.  On sparse data -- a toy with three distinct
    codes, where smoothing dominates -- the fixed convention drove the coefficient
    from 16.7 up to 32.3 as the space shrank, while the fixed-total convention
    held it at 16.7--17.0 before dropping to 6.6 at the most extreme coarsening.
    On realistic density (40,000 samples over 1296 cells) both conventions are
    monotone non-increasing across seven coarsening steps, and they differ only in
    range: 2.60 to 2.28 fixed-per-state against 2.60 to 2.19 fixed-total.  The
    population quantity obeys the data-processing inequality; the smoothed
    estimator obeys it too once the counts are dense, because the smoothing is
    applied after coarsening and so is not itself a coarsening.

    The per-state default is kept so that values already recorded stay
    reproducible; any comparison across granularities must pass ``total_alpha``.
    """
    if total_alpha is not None:
        alpha = total_alpha / max(n_states, 1)
    d = histogram(d_codes, n_states) + alpha
    m = histogram(mu_codes, n_states) + alpha
    d /= d.sum()
    m /= m.sum()
    ratio = d / m
    support = d > 0
    c2_raw = float((d[support] * ratio[support]).sum())
    c2_clip = float((d[support] * np.minimum(ratio[support], clip)).sum())
    # Fraction of the teacher's mass sitting where mu is thinner by >clip.
    heavy = float(d[support][ratio[support] > clip].sum())
    return {"c2": c2_clip, "c2_unclipped": c2_raw, "mass_above_clip": heavy,
            "n_states_seen_d": int((histogram(d_codes, n_states) > 0).sum()),
            "n_states_seen_mu": int((histogram(mu_codes, n_states) > 0).sum()),
            "n_d": len(d_codes), "n_mu": len(mu_codes), "alpha": alpha,
            "clip": clip, "estimator": "A_counting"}


# --------------------------------------------------------------- estimator B


def c2_density_ratio(d_feat: np.ndarray, mu_feat: np.ndarray,
                     folds: int = 2, clip: float = CLIP_M,
                     pca: int | None = 64, seed: int = 0) -> dict:
    r"""C2 from a two-sample classifier, plus the AUC that makes it auditable.

    With balanced classes the Bayes-optimal logistic score satisfies
    ``g/(1-g) = d(h)/mu(h)``, so the classifier *is* the density-ratio
    estimator.  Balance is enforced by subsampling rather than corrected for,
    because the correction is a log-odds shift that is easy to get wrong and
    impossible to notice.

    Cross-fitting is not optional.  The ratio is evaluated only on rows the
    fold's classifier never saw; in-sample ratios are overconfident and inflate
    C2 without bound.  The returned AUC is the diagnostic for gate G2: at
    AUC 0.5 the two measures are indistinguishable in this feature space and
    there is nothing for the method to fix.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = min(len(d_feat), len(mu_feat))
    if n < 50:
        return {"c2": float("nan"), "auc": float("nan"), "n": n,
                "estimator": "B_density_ratio", "note": "too few samples"}
    di = rng.permutation(len(d_feat))[:n]
    mi = rng.permutation(len(mu_feat))[:n]
    X = np.concatenate([d_feat[di], mu_feat[mi]]).astype(np.float64)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    if pca and X.shape[1] > pca:
        X = PCA(n_components=pca, random_state=seed).fit_transform(X)

    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    w = np.zeros(len(y))
    auc_parts = []
    bounds = np.linspace(0, len(y), folds + 1).astype(int)
    for k in range(folds):
        te = np.zeros(len(y), bool)
        te[bounds[k]:bounds[k + 1]] = True
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[~te], y[~te])
        p = clf.predict_proba(X[te])[:, 1].clip(1e-6, 1 - 1e-6)
        w[te] = p / (1.0 - p)
        if 0 < y[te].sum() < te.sum():
            auc_parts.append(roc_auc_score(y[te], p))
    w = np.minimum(w, clip)
    return {"c2": float(w[y == 1].mean()), "auc": float(np.mean(auc_parts)),
            "auc_folds": [float(a) for a in auc_parts], "n": n,
            "clip": clip, "estimator": "B_density_ratio"}


# --------------------------------------------------------------- estimator C


def c2_knn(d_feat: np.ndarray, mu_feat: np.ndarray, k: int = 5,
           dim: int = 16, clip: float = CLIP_M, seed: int = 0) -> dict:
    r"""C2 from k-nearest-neighbour density ratios in a low-dimensional space.

    The ratio of k-NN radii estimates ``d/mu`` up to the sample-size factor:
    ``w(x) = (n_mu / n_d) * (r_mu(x) / r_d(x))^dim``.  Raising a distance ratio
    to the power of the dimension is why this estimator is only used after
    reducing to ``dim = 16`` -- and why it is the weakest of the three and is
    reported, never relied upon.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n = min(len(d_feat), len(mu_feat))
    if n < 50:
        return {"c2": float("nan"), "n": n, "estimator": "C_knn",
                "note": "too few samples"}
    di = rng.permutation(len(d_feat))[:n]
    mi = rng.permutation(len(mu_feat))[:n]
    X = np.concatenate([d_feat[di], mu_feat[mi]]).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    if X.shape[1] > dim:
        X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    A, B = X[:n], X[n:]
    # r_d excludes the query itself (k+1 neighbours, drop the zero distance).
    rd = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A)[0][:, k]
    rm = NearestNeighbors(n_neighbors=k).fit(B).kneighbors(A)[0][:, k - 1]
    eps = 1e-12
    w = np.minimum(((rm + eps) / (rd + eps)) ** dim, clip)
    return {"c2": float(w.mean()), "k": k, "dim": dim, "n": n, "clip": clip,
            "estimator": "C_knn"}


# --------------------------------------------------------------- convenience


def all_estimators(d_codes, mu_codes, n_states, d_feat=None, mu_feat=None,
                   seed: int = 0) -> dict:
    out = {"A": c2_counting(d_codes, mu_codes, n_states)}
    if d_feat is not None and mu_feat is not None and len(d_feat) and len(mu_feat):
        out["B"] = c2_density_ratio(d_feat, mu_feat, seed=seed)
        out["C"] = c2_knn(d_feat, mu_feat, seed=seed)
    return out


def select_prefixes_by_ratio(d_codes: Sequence[int], mu_codes: Sequence[int],
                             cand_codes: Sequence[int], n_states: int,
                             budget: int, alpha: float = 0.5,
                             seed: int = 0) -> np.ndarray:
    r"""Proposition 2: expand where ``d(h) * d(h)/mu(h)`` is largest.

    Minimising ``C2`` over where to spend a fixed number of teacher calls gives,
    to first order, a rule proportional to ``d * w`` rather than to ``w`` alone:
    a state the teacher almost never visits is not worth a call however thin the
    supervision is there.  That factor is the whole difference between this rule
    and "sample the rarest thing you can find", and it is why the random-prefix
    control in gate G5 is a real control rather than a straw man.

    Ties are broken at random with a fixed seed, and selection is *with* replace-
    ment across states but without replacement across candidate prefixes.
    """
    d = histogram(d_codes, n_states) + alpha
    m = histogram(mu_codes, n_states) + alpha
    d /= d.sum()
    m /= m.sum()
    score = d * np.minimum(d / m, CLIP_M)
    rng = np.random.default_rng(seed)
    cand = np.asarray(cand_codes, dtype=np.int64)
    key = score[cand] + rng.uniform(0, 1e-9, size=len(cand))
    return np.argsort(-key)[:budget]
