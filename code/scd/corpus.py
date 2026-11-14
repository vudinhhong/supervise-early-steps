r"""The distillation corpus, and the supervision measure defined on it.

A corpus row is a trace plus a *step mask*: which reasoning steps carry a loss.
The measure ``mu`` is then the multiset of abstraction codes at the supervised step
boundaries, and it is what the coverage estimators consume.  Keeping the mask and
the measure in one object is what makes gate G3 possible: three arms can share
byte-identical text and differ only in the mask.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from . import abstraction, steps


@dataclass
class Row:
    question: str
    gold: str
    steps: List[str]
    correct: bool
    source: str = ""
    origin: str = "teacher"        # teacher | student | branch
    branch_at: int = -1
    step_mask: List[bool] = field(default_factory=list)

    def sigma(self) -> List[int]:
        return abstraction.sigma_path(self.question, self.steps)

    def supervised_codes(self) -> List[int]:
        path = self.sigma()
        mask = self.step_mask or [True] * len(self.steps)
        # boundary t is supervised iff step t (1-indexed) carries a loss
        return [path[t + 1] for t, on in enumerate(mask) if on]

    def n_supervised_steps(self) -> int:
        return sum(self.step_mask) if self.step_mask else len(self.steps)


def from_records(recs: Iterable[dict], origin: str = "teacher") -> List[Row]:
    out = []
    for r in recs:
        out.append(Row(question=r["question"], gold=r["gold"], steps=list(r["steps"]),
                       correct=bool(r.get("correct", False)),
                       source=r.get("source", ""), origin=origin,
                       branch_at=int(r.get("branch_at", -1))))
    return out


def save(rows: Sequence[Row], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__) + "\n")


def load(path: pathlib.Path) -> List[Row]:
    return [Row(**json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


# ------------------------------------------------------- measure manipulations


def _set_mask(rows: Sequence[Row], fn) -> None:
    for r in rows:
        r.step_mask = [bool(x) for x in fn(r)]


def mask_all(rows: Sequence[Row]) -> None:
    _set_mask(rows, lambda r: [True] * len(r.steps))


def mask_fraction(rows: Sequence[Row], frac: float, mode: str,
                  seed: int = 0) -> None:
    r"""Supervise a ``frac`` share of steps, chosen by ``mode``.

    ``mode`` is the independent variable of gate G3.  ``uniform`` spreads the
    supervised positions over the whole trace; ``early`` and ``late`` concentrate
    them; ``cluster_drop`` bans cells rarest-first until ``1 - frac`` of the count
    *mass* is gone, which at frac = 0.25 leaves a single cell -- the most extreme
    point on the coverage axis, and more extreme than the name suggests;
    ``cell_drop`` instead drops a random ``1 - frac`` of the *cells*, leaving
    partial coverage, which is what a test of *where* to spend a budget needs.

    The unsupervised steps stay in the context -- only the loss is masked.
    Truncating the trace instead would change the input as well as the measure,
    and the experiment would no longer isolate anything (validity condition V4.2).
    """
    rng = np.random.default_rng(seed)
    if mode in ("uniform", "early", "late"):
        def fn(r):
            m = len(r.steps)
            k = max(1, int(round(frac * m)))
            if mode == "early":
                idx = range(k)
            elif mode == "late":
                idx = range(m - k, m)
            else:
                idx = rng.choice(m, size=k, replace=False)
            out = [False] * m
            for i in idx:
                out[int(i)] = True
            return out
        _set_mask(rows, fn)
        return
    if mode == "cell_drop":
        # Drop a fraction of CELLS, chosen at random, keeping the rest.
        #
        # This exists because ``cluster_drop`` is not the manipulation its name
        # suggests.  That mode bans the rarest cells until ``1 - frac`` of the
        # count *mass* is gone, and the cell distribution is steep enough that at
        # frac = 0.25 it leaves exactly **one** cell of the 413 the teacher visits.
        # Stage G6 ran on that measure for three and a half GPU-hours and could not
        # mean anything: from a single cell, any way of adding four thousand
        # diverse prefixes covers three to four hundred cells, so *which* prefixes
        # the rule picks stops mattering -- random scored 1.797 against the
        # ratio rule's 1.912, and the two are the same answer.
        #
        # A test of where to spend a budget needs supervision that is partially
        # deficient, not annihilated.
        counts: Dict[int, int] = {}
        for r in rows:
            for c in r.sigma()[1:]:
                counts[c] = counts.get(c, 0) + 1
        cells = sorted(counts)
        n_drop = int(round((1.0 - frac) * len(cells)))
        banned = set(rng.choice(cells, size=min(n_drop, len(cells)),
                                replace=False).tolist()) if n_drop > 0 else set()
        _set_mask(rows, lambda r: [c not in banned for c in r.sigma()[1:]])
        return
    if mode != "cluster_drop":
        raise ValueError(mode)

    counts: Dict[int, int] = {}
    for r in rows:
        for c in r.sigma()[1:]:
            counts[c] = counts.get(c, 0) + 1
    order = sorted(counts, key=lambda c: counts[c])       # rarest cells first
    target_drop = (1.0 - frac) * sum(counts.values())
    dropped, banned = 0, set()
    for c in order:
        if dropped >= target_drop:
            break
        banned.add(c); dropped += counts[c]
    _set_mask(rows, lambda r: [c not in banned for c in r.sigma()[1:]])


def match_supervised_count(rows: Sequence[Row], target: int,
                           seed: int = 0) -> int:
    r"""Turn supervised steps off at random until the total equals ``target``.

    Validity condition V4.1: three measures compared at different numbers of
    supervised positions are a data-quantity experiment wearing a coverage
    experiment's clothes.  Returns the achieved total.
    """
    rng = np.random.default_rng(seed)
    have = sum(r.n_supervised_steps() for r in rows)
    if have <= target:
        return have
    on = [(i, t) for i, r in enumerate(rows)
          for t, v in enumerate(r.step_mask or [True] * len(r.steps)) if v]
    for i, r in enumerate(rows):
        if not r.step_mask:
            r.step_mask = [True] * len(r.steps)
    drop = rng.choice(len(on), size=have - target, replace=False)
    for j in drop:
        i, t = on[int(j)]
        rows[i].step_mask[t] = False
    return sum(r.n_supervised_steps() for r in rows)


def measure_codes(rows: Sequence[Row]) -> List[int]:
    return [c for r in rows for c in r.supervised_codes()]


def teacher_codes(rows: Sequence[Row]) -> List[int]:
    """``d`` -- every boundary of every teacher trace, mask ignored."""
    return [c for r in rows if r.origin == "teacher" for c in r.sigma()[1:]]
