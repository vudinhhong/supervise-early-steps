r"""The state abstraction $\sigma$, and why the study rests on it.

Estimator B of the coverage coefficient reads the student's hidden state at a
prefix boundary and asks a classifier to tell teacher prefixes from supervision
prefixes.  That estimator has a fatal confound available to it: it can answer
by recognising *who wrote the text* instead of *which reasoning state the text
reached*.  The earlier compute-matched study was killed by exactly this kind of
leak -- its per-token demand signal reached AUC 0.82, and a control that saw
only token log-frequency already reached 0.61.

$\sigma$ removes the confound by construction rather than by controlling for it.
It maps a prefix to a triple

    (step bucket, magnitude bucket of the newest derived number, operation type)

so two prefixes that reach the same arithmetic state by different wording are
*the same state*.  Paraphrase invariance is then not an empirical hope tested in
gate G2 -- it is a property of the function.  Two consequences:

1. The coefficient computed on $\sigma$ is a lower bound on the true one, by the
   data-processing inequality for $\chi^2$: coarsening cannot increase
   divergence.  We therefore always under-report coverage damage.  A large value
   is trustworthy; a value near 1 is *not* evidence that coverage is fine.
2. The abstraction is arithmetic, so it is faithful only where reasoning states
   *are* arithmetic.  Scope is mathematics and code, and the manuscript says so
   instead of claiming distillation in general.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple

N_STEP_B = 6          # 0,1,2,3,4,>=5
N_MAG_B = 24          # 11 magnitudes x 2 signs, plus zero, plus "no new number"
N_OP_B = 9
N_STATES = N_STEP_B * N_MAG_B * N_OP_B        # 1296

OPS = ("add", "sub", "mul", "div", "mod", "cmp", "restate", "other", "none")
_OP_INDEX = {o: i for i, o in enumerate(OPS)}

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")


MAX_DIGITS = 24        # see tasks.MAX_ANSWER_DIGITS; same failure, same guard


def numbers(text: str) -> List[float]:
    """Finite numbers in reading order, with LaTeX fractions evaluated.

    Non-finite values must never escape this function.  ``_magnitude_bucket``
    takes ``log10`` of what it is given, and ``int(floor(inf))`` raises -- so a
    single degenerate digit run in one trace would abort a whole coverage
    measurement rather than costing one state.
    """
    text = _FRAC.sub(lambda m: _safe_div(m.group(1), m.group(2)), text)
    out = []
    for s in _NUM.findall(text):
        if sum(c.isdigit() for c in s) > MAX_DIGITS:
            continue
        try:
            v = float(s.replace(",", ""))
        except (ValueError, OverflowError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _safe_div(a: str, b: str) -> str:
    try:
        return f" {float(a) / float(b)} "
    except Exception:
        return f" {a} {b} "


def _magnitude_bucket(v: float | None) -> int:
    """0 = no new number, 1 = exactly zero, 2..12 = positive, 13..23 = negative.

    The zero code must not share an index with the smallest positive bucket:
    a first draft wrote ``1 + (e + 2)`` for positives, which collides with the
    zero code at ``e = -2`` and silently merges two states.
    """
    if v is None or not math.isfinite(v):
        return 0
    if v == 0:
        return 1
    e = max(-2, min(8, int(math.floor(math.log10(abs(v))))))   # 11 values
    return (2 + (e + 2)) if v > 0 else (13 + (e + 2))


def _op_of(step: str) -> str:
    s = step
    if re.search(r"(?<![<>=!])=(?![=])", s) and re.search(r"[<>]|at least|at most|more than|less than", s):
        return "cmp"
    if "%" in s or " mod " in s or "remainder" in s:
        return "mod"
    if re.search(r"[*×x]\s*\d|\d\s*[*×]|times|product|multipl", s):
        return "mul"
    if re.search(r"[/÷]\s*\d|divide|per |quotient|half|third", s):
        return "div"
    if re.search(r"-\s*\d|minus|subtract|left|remain|differen|fewer", s):
        return "sub"
    if re.search(r"\+|plus|add|total|sum|altogether|combined", s):
        return "add"
    if re.search(r"[<>]|greater|smaller|compare", s):
        return "cmp"
    if not _NUM.search(s):
        return "restate"
    return "other"


def encode(step_index: int, prev_numbers: Sequence[float],
           new_numbers: Sequence[float], step_text: str) -> int:
    """Index in ``[0, N_STATES)``.

    ``new_numbers`` are the numbers introduced by the last step that did not
    already occur in the question or in earlier steps -- the newest *derived*
    quantity.  Numbers copied out of the question carry no reasoning state and
    are excluded, which is what makes the code below care about ``prev``.
    """
    t_b = min(step_index, N_STEP_B - 1)
    seen = set(round(v, 6) for v in prev_numbers)
    fresh = [v for v in new_numbers if round(v, 6) not in seen]
    v_b = _magnitude_bucket(fresh[-1] if fresh else None)
    o_b = _OP_INDEX["none"] if step_index == 0 else _OP_INDEX[_op_of(step_text)]
    return (t_b * N_MAG_B + v_b) * N_OP_B + o_b


def sigma_path(question: str, steps: Sequence[str]) -> List[int]:
    """One state per prefix, for prefixes of length ``0..len(steps)``.

    Returned length is ``len(steps) + 1``.  The empty prefix is a state too: it
    is where every trace starts, and a supervision measure that never scores the
    empty prefix is missing the position at which the student commits to an
    approach.
    """
    seen = list(numbers(question))
    path = [encode(0, seen, [], "")]
    for t, st in enumerate(steps):
        new = numbers(st)
        path.append(encode(t + 1, seen, new, st))
        seen.extend(new)
    return path


def describe(code: int) -> str:
    o_b = code % N_OP_B
    v_b = (code // N_OP_B) % N_MAG_B
    t_b = code // (N_OP_B * N_MAG_B)
    if v_b == 0:
        mag = "no-new"
    elif v_b == 1:
        mag = "0"
    elif v_b <= 12:
        mag = f"+1e{v_b - 4}"
    else:
        mag = f"-1e{v_b - 15}"
    return f"t{'>=5' if t_b == 5 else t_b}/{mag}/{OPS[o_b]}"


def recode(codes, mag_bins: int):
    r"""Re-express codes on a coarser magnitude axis: ``(new_codes, n_states)``.

    This exists because the sensitivity check it serves was vacuous.  Gate G2.7
    asks whether ``C2`` survives changing the number of magnitude buckets, and the
    stage looped over ``--mag-bins`` while passing the module-level ``N_STATES``
    to every call -- so all the bin settings produced byte-identical codes, the
    coefficient of variation across them was exactly zero, and the criterion could
    not fail.  A criterion that cannot fail is worse than no criterion: it reads as
    evidence.

    Decoding and re-encoding the triple is exact and avoids re-parsing any text,
    so the check costs nothing.  ``mag_bins`` of 1 collapses magnitude entirely,
    which is the strongest coarsening worth asking about.
    """
    import numpy as np
    c = np.asarray(codes, dtype=np.int64)
    o_b = c % N_OP_B
    v_b = (c // N_OP_B) % N_MAG_B
    t_b = c // (N_OP_B * N_MAG_B)
    nb = max(1, min(int(mag_bins), N_MAG_B))
    v_new = (v_b * nb) // N_MAG_B
    return ((t_b * nb + v_new) * N_OP_B + o_b), N_STEP_B * nb * N_OP_B
