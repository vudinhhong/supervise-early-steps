r"""An exactly solvable world, so gates G0 and G1 cost CPU and not GPU.

Everything the theory refers to -- the teacher's state occupancy, the supervision
measure, the fidelity term, the coverage coefficient, and the true capability gap
-- is computed here by enumeration or by exact dynamic programming, with no
estimation anywhere.  Three questions become answerable before any model trains:

  G0.0  does the world reproduce the phenomenon under study at all?
  G0    do the sample-based estimators of ``C2`` recover a known ``C2``?
  G1    does the product of the two factors predict the true capability gap
        better than the fidelity factor alone?

G1 is the gate the earlier tangent-space study failed after five months of GPU
work.  Here the same question costs a day of CPU.


The world
---------

A first-order chain on ``NS`` latent states, horizon ``H``.  The task is a hidden
grammar: at each state a subset of actions is forbidden.  The objective is the
*rate* of grammatical steps,

    J(pi) = sum_s d^pi(s) sum_a pi(a|s) r(s,a),    r = "action is allowed",

which is the average-reward form the performance-difference lemma is stated for,
is bounded in ``[0, 1]`` at every horizon, and depends on the student's own
occupancy as well as its conditionals -- so distribution shift is in the objective
rather than assumed away.  The teacher respects the grammar, so ``J(teacher) = 1``
exactly; a policy that ignores it scores the chance rate ``1 - n_forbidden/na``.

An all-or-nothing reward ("the whole rollout is grammatical") was tried first and
discarded: at ``H = 40`` every imperfect student scores 0 and the dependent
variable has no variance left to explain.  That is the same defect as the unlearnable
first task, in a different disguise, and it is why G0.0 checks dynamic range
explicitly instead of trusting the design.

Capacity is *parameter tying*: the student must answer identically at every state
inside a randomly drawn group, so ``k = NS`` groups is a student that can match
the teacher and ``k = 1`` is a student with one context-free distribution.
Coverage is the supervision measure ``mu`` over states.  The student is then the
exact forward-KL projection of the teacher onto ``mu``-weighted groups -- so the
student is a *function of* ``mu``, which is what training is.

Two design errors are recorded here because the protocol exists to catch this
class of mistake, and both were caught before any GPU time.

*The reward must be learnable.*  A first version rewarded
``y_H == (y_1 + ... + y_{H-1}) mod V``, which no Markov policy can do, so every
policy scored about ``1/V`` and the dependent variable had a range of 0.013.  Gate
G1 would have reported a weak correlation and killed the direction -- not because
the theory is wrong but because the assay had no dynamic range.  That is the
synthetic analogue of gate GP, and it is now checked as G0.0(a).

*The student must depend on ``mu``.*  A first version perturbed the teacher at
random to make a student, so the true capability gap did not depend on ``mu`` at
all, and a gate asking whether coverage predicts the gap was meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

NS_DEFAULT = 64
NA_DEFAULT = 8
SMOOTH = 1e-4             # what an ESTIMATOR does to mu; the world does not


# --------------------------------------------------------------------- world


@dataclass
class World:
    """Transition structure, grammar, and the teacher, all fixed together."""
    ns: int
    na: int
    H: int
    nxt: np.ndarray            # (ns, na) -> next state
    allowed: np.ndarray        # (ns, na) bool
    pi_t: np.ndarray           # (ns, na) teacher, respects the grammar
    s0: int = 0


def make_world(seed: int, ns: int = NS_DEFAULT, na: int = NA_DEFAULT,
               H: int = 40, n_forbidden: int = 5, alpha: float = 0.3) -> World:
    rng = np.random.default_rng(seed)
    nxt = rng.integers(0, ns, size=(ns, na))
    allowed = np.ones((ns, na), bool)
    for s in range(ns):
        allowed[s, rng.choice(na, size=n_forbidden, replace=False)] = False
    pi = rng.dirichlet(np.full(na, alpha), size=ns) * allowed
    pi = np.where(pi.sum(1, keepdims=True) > 0, pi, allowed.astype(float))
    pi /= pi.sum(1, keepdims=True)
    return World(ns=ns, na=na, H=H, nxt=nxt, allowed=allowed, pi_t=pi)


# ------------------------------------------------------- exact state measures


def occupancy(w: World, pi: np.ndarray, grammatical: bool = True) -> np.ndarray:
    r"""``d(s) = (1/H) sum_t Pr[s_t = s]``, exactly, by forward recursion.

    ``grammatical=True`` follows only allowed actions, which is the right measure
    for the teacher: it is the distribution over states the teacher actually
    visits while solving the task.  ``False`` follows the policy as written, which
    is the right measure for a student that violates the grammar.
    """
    p = np.zeros(w.ns); p[w.s0] = 1.0
    acc = np.zeros(w.ns)
    trans = pi * (w.allowed if grammatical else 1.0)
    for _ in range(w.H):
        acc += p
        nxt = np.zeros(w.ns)
        for a in range(w.na):
            np.add.at(nxt, w.nxt[:, a], p * trans[:, a])
        tot = nxt.sum()
        p = nxt / tot if tot > 0 else np.zeros(w.ns)
    return acc / max(acc.sum(), 1e-300)


def step_reward(w: World, pi: np.ndarray) -> np.ndarray:
    """``rho(s) = sum_a pi(a|s) r(s,a)``: the per-visit grammaticality of ``pi``."""
    return (pi * w.allowed).sum(1)


def value(w: World, pi: np.ndarray) -> float:
    """``J(pi)``: the rate of grammatical steps, exactly.

    The occupancy is the policy's *own*, unrestricted by the grammar -- a student
    that takes a forbidden action keeps going and ends up in states the teacher
    never visits.  That is the compounding channel the coverage term is supposed
    to be about, so removing it would beg the question.
    """
    d = occupancy(w, pi, grammatical=False)
    return float((d * step_reward(w, pi)).sum())


def bound(w: World, c2: float, e_kl: float) -> float:
    r"""Theorem 1 in the average-reward normalisation.

        gap <= sqrt(2) * H * sqrt( C2(mu) * E_mu[KL] )

    Derivation.  The performance-difference lemma over ``H`` steps with
    ``|Q| <= H`` gives ``H * gap <= 2 H^2 E_d[TV]``; Cauchy--Schwarz moves the
    expectation from ``d`` to ``mu`` at the cost of ``sqrt(C2)``; Pinsker turns
    ``E_mu[TV^2]`` into ``E_mu[KL]/2``.

    **Both arguments must be taken under the same measure.**  A first version
    passed ``C2`` computed under a smoothed ``mu`` and ``E_mu[KL]`` computed under
    the raw one, and reported 22 violations of a theorem in 260 configurations --
    which is not a counterexample, it is an inconsistent pair.  The failure is
    instructive rather than embarrassing: when the raw ``mu`` has a hole, the
    limiting pair is ``(C2 = infinity, E_mu[KL] = 0)`` and the bound reads
    ``gap <= 0 * infinity``.  That vacuity *is* Corollary 1.1 -- fidelity measured
    on a measure with a hole carries no information about capability -- so the
    right response is to report the bound under a single smoothed measure and to
    report the vacuous raw pair separately, as evidence.
    """
    return 2.0 ** 0.5 * w.H * (max(c2, 0.0) * max(e_kl, 0.0)) ** 0.5


def kl_per_state(w: World, pi_s: np.ndarray) -> np.ndarray:
    p, q = w.pi_t, pi_s
    return (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(1)


def c2_exact(d: np.ndarray, mu: np.ndarray) -> float:
    m = np.maximum(mu, 1e-300)
    return float((d * d / m)[d > 0].sum())


def coarsen(x: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(codes, weights=x, minlength=k)


# -------------------------------------------------------------- the student


def capacity_groups(ns: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A random partition of states into ``k`` groups -- the student's capacity.

    Drawn, never chosen.  Validity condition V1.2: groups arranged to merge
    exactly the states the supervision misses would make gate G1 pass by
    construction and prove nothing.
    """
    return rng.permutation(np.arange(ns) % k)


def student_from_mu(w: World, mu: np.ndarray, groups: np.ndarray,
                    fallback: str = "uniform") -> np.ndarray:
    r"""The student forward-KL distillation on ``mu`` converges to, in closed form.

    Minimising ``E_{s ~ mu} KL(pi_T(.|s) || pi_S(.|s))`` subject to ``pi_S`` being
    constant on each group is a weighted M-projection, solved by the ``mu``-weighted
    mixture of the teacher's rows in the group.  Two consequences drive the study:

    *A narrow ``mu`` can give a lower training loss and a worse student.*  It fits
    the groups it sees and is left at ``fallback`` elsewhere.

    *An unsupervised group keeps ``fallback``.*  ``uniform`` is the neutral default
    for "no signal"; ``teacher_marginal`` is the ablation standing for a student
    that begins from a pretrained prior rather than from nothing.
    """
    back = (np.full(w.na, 1.0 / w.na) if fallback == "uniform"
            else w.pi_t.mean(0) / w.pi_t.mean(0).sum())
    out = np.empty_like(w.pi_t)
    for g in range(int(groups.max()) + 1):
        rows = np.flatnonzero(groups == g)
        if not len(rows):
            continue
        tot = mu[rows].sum()
        out[rows] = back if tot <= 0 else (mu[rows, None] * w.pi_t[rows]).sum(0) / tot
    return out / out.sum(1, keepdims=True)


# ---------------------------------------------------------- mu constructions


def mu_family(w: World, seed: int = 0) -> Dict[str, np.ndarray]:
    r"""Supervision measures spanning ``C2`` over roughly ``[1, 100]``.

    Two shapes, deliberately:

    ``mix``   ``mu = (1 - lam) d + lam nu`` bounds ``C2 <= 1/(1-lam)`` and is what
              a distillation corpus that contains *some* on-distribution states
              looks like.

    ``hole``  ``mu`` is zero on a set of states that ``d`` reaches.  This is the
              regime Corollary 1.1 is actually about, and the regime the mixture
              family can never enter.  A first draft used mixtures only, capped
              ``C2`` at 50, and therefore tested the theory nowhere near the place
              it makes its strongest claim.

    Bases are chosen for variety of shape, not to favour the hypothesis: other
    policies' occupancies, a uniform measure, an early/late split of the horizon,
    and random state subsets.
    """
    rng = np.random.default_rng(seed)
    d = occupancy(w, w.pi_t)
    uni = np.ones(w.ns) / w.ns

    def norm(x):
        x = np.maximum(x, 0.0)
        s = x.sum()
        return x / s if s > 0 else uni.copy()

    other = make_world(seed + 9991, ns=w.ns, na=w.na, H=w.H)
    bases = {"uniform": uni.copy(),
             "third": norm(occupancy(w, other.pi_t, grammatical=False)),
             "greedy": norm(occupancy(w, _onehot(w.pi_t.argmax(1), w.na),
                                      grammatical=False))}
    out: Dict[str, np.ndarray] = {"identity": norm(d)}
    for name, lam in [("uniform", 0.5), ("uniform", 0.9), ("uniform", 0.99),
                      ("third", 0.5), ("third", 0.9), ("third", 0.99),
                      ("greedy", 0.9)]:
        out[f"mix:{name}@{lam}"] = norm((1 - lam) * d + lam * bases[name])
    for frac in (0.25, 0.5, 0.75):
        keep = rng.random(w.ns) >= frac
        m = d * keep
        if m.sum() > 0:
            out[f"hole:{frac}"] = norm(m)
    order = np.argsort(-d)                     # keep only the most-visited states
    for frac in (0.25, 0.5):
        m = np.zeros(w.ns); take = order[:max(1, int(w.ns * frac))]
        m[take] = d[take]
        out[f"head:{frac}"] = norm(m)
    return out


def smoothed(mu: np.ndarray, ns: int, smooth: float = SMOOTH) -> np.ndarray:
    r"""What an *estimator* sees: ``mu`` with additive smoothing.

    The distinction matters and getting it wrong silently deletes the regime the
    theory is strongest about.  A first draft added the floor to ``mu`` itself,
    which meant every state carried supervision, no capacity group ever fell back
    to its prior, and at full capacity the student recovered the teacher exactly
    under *every* supervision measure -- the dependent variable had zero variance
    in the one regime Corollary 1.1 describes.

    So: the world keeps the true zeros, and only the coverage *estimate* smooths.
    That is also what happens in practice -- a corpus really does omit states, and
    a finite-sample estimator really does add counts.
    """
    uni = np.ones(ns) / ns
    return (1 - smooth) * (mu / max(mu.sum(), 1e-300)) + smooth * uni


def _onehot(idx: np.ndarray, na: int) -> np.ndarray:
    m = np.zeros((len(idx), na)); m[np.arange(len(idx)), idx] = 1.0
    return m


# ------------------------------------------------------------------ sampling


def sample_states(measure: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(measure), size=n, p=measure / measure.sum())


def sigma_codes(ns: int, k: int, seed: int = 7) -> np.ndarray:
    """A coarsening for estimator A: a fixed random partition into ``k`` cells.

    Coarsening cannot increase ``chi^2``, so estimator A returns a lower bound on
    the exact coefficient.  Gate G0 therefore checks *accuracy* against the
    exactly computable coarsened value and *ranking* against the full one:
    blaming an estimator for a property of its feature map is a mistake that
    would be easy to make here and impossible to see later.
    """
    return np.random.default_rng(seed).permutation(np.arange(ns) % k)


def features(state_idx: np.ndarray, ns: int) -> np.ndarray:
    """One-hot state identity -- the richest feature map available, no coarsening."""
    F = np.zeros((len(state_idx), ns), dtype=np.float32)
    F[np.arange(len(state_idx)), state_idx] = 1.0
    return F
