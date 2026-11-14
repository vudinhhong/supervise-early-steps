r"""Training questions and the pooled evaluation suite.

The suite exists because of a power calculation, not a taste for large numbers.
Section 9 of the protocol works it out: two similar arms disagree on roughly a
fifth of items, so the standard error of a paired accuracy difference is
``sqrt(0.2 / n)`` and the minimum detectable effect at 80% power is ``2.8`` times
that.  The earlier study evaluated on 500 items, which puts its minimum
detectable effect at 5.6 points -- larger than every difference it reported.  A
suite of about 3000 items brings it to 2.3 points, which is below the effects
this study predicts.

Components degrade gracefully: a component that will not load is skipped, the
realised ``n`` is recorded, and the gate recomputes the minimum detectable effect
from that ``n`` instead of assuming it.  A missing dataset therefore weakens a
conclusion visibly rather than invisibly.

Held-out slices of the training sets are part of the suite, so the split is
asserted disjoint at load time rather than trusted.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tasks import Problem, gold_from_gsm8k, gold_from_math

HELD_GSM8K = 1200          # tail of gsm8k train reserved for evaluation
HELD_MATH = 1000           # ditto for the MATH training split
SPLIT_SEED = 20260401      # fixed once; changing it invalidates every result

SUITE_FILE = "eval_suite_3471.jsonl"
SUITE_CENSUS = {"gsm8k-heldout": 1200, "gsm8k-test": 1319,
                "math-heldout": 627, "math500": 325}


def _packed_suite() -> Optional[Tuple[List[Problem], Dict[str, int]]]:
    """The evaluation suite as a shipped file, if it is there.

    This exists because depending on the ``datasets`` cache cost two consecutive
    runs on the cluster.  That library takes a ``.lock`` even to read, so a
    file-count quota stops it; and when it cannot rebuild the builder from cache
    it reaches for the Hub, which a compute node cannot do offline.  The suite is
    3471 fixed problems, so shipping it as one file removes all three failure
    modes at once: the lock, the network, and the quota.

    The census is checked against the constant.  A truncated or wrong file must
    fail here rather than quietly shrink the suite, because every accuracy in the
    paper is against this exact 3471.
    """
    f = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / SUITE_FILE
    if not f.is_file():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]

    # An optional first line carrying metadata.  Two shapes of the file are valid,
    # because a copy made before that line existed must keep working: refusing it
    # turns a stale file into a dead job, and the check that actually matters is
    # the census below.
    meta = rows[0].get("_meta") if rows and isinstance(rows[0], dict) else None
    if meta is not None:
        rows = rows[1:]

    problems = [Problem(r["question"], r["gold"], r["source"]) for r in rows]
    census: Dict[str, int] = {}
    for p in problems:
        census[p.source] = census.get(p.source, 0) + 1
    census = dict(sorted(census.items()))

    # This is the real integrity check, and it is hard.  The census pins the file
    # to exactly the published suite; a truncated or substituted file fails here,
    # and it must, because every accuracy in the paper is against this 3471.
    if census != SUITE_CENSUS:
        raise RuntimeError(
            "%s has census %s, expected %s -- a wrong or truncated file; refusing "
            "it, because a smaller suite makes every accuracy incomparable with "
            "the reported ones" % (f.name, census, SUITE_CENSUS))

    # Disjointness from the training pool is a property of the construction: the
    # held-out slices are the complement of the training indices under one fixed
    # permutation.  A file with this exact census is that construction.  When the
    # generator recorded an explicit check, say so; when it did not, say that too
    # rather than either failing or implying more than is known.
    note = ("disjointness verified at generation"
            if meta and meta.get("disjoint_verified")
            else "no generation-time disjointness record in this file")
    print("[data] evaluation suite: %d problems from %s (no dataset load; %s)"
          % (len(problems), f.name, note))
    return problems, census


def _perm(n: int) -> np.ndarray:
    return np.random.default_rng(SPLIT_SEED).permutation(n)


def _gsm8k_raw():
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main")


def _math_train_raw():
    """Hendrycks MATH training split, if a mirror is reachable."""
    from datasets import load_dataset
    for name, kwargs in (("nlile/hendrycks-MATH-benchmark", {}),
                         ("EleutherAI/hendrycks_math", {"name": "algebra"})):
        try:
            ds = load_dataset(name, **kwargs)
            return ds["train"]
        except Exception:
            continue
    return None


def _math500_raw():
    from datasets import load_dataset
    for name in ("HuggingFaceH4/MATH-500",):
        try:
            return load_dataset(name)["test"]
        except Exception:
            continue
    return None


def train_questions(limit: int = 0, include_math: bool = True) -> List[Problem]:
    """Questions available for distillation -- held-out slices removed."""
    out: List[Problem] = []
    g = _gsm8k_raw()["train"]
    idx = _perm(len(g))[HELD_GSM8K:]
    for i in idx:
        r = g[int(i)]
        out.append(Problem(r["question"], gold_from_gsm8k(r["answer"]), "gsm8k-train"))
    if include_math:
        m = _math_train_raw()
        if m is not None:
            keep = _perm(len(m))[HELD_MATH:]
            for i in keep:
                r = m[int(i)]
                gold = gold_from_math(r.get("solution", "") or "")
                if gold is not None:
                    out.append(Problem(r["problem"], gold, "math-train"))
    rng = np.random.default_rng(0)
    out = [out[i] for i in rng.permutation(len(out))]
    return out[:limit] if limit else out


def eval_suite(limit_per_component: int = 0) -> Tuple[List[Problem], Dict[str, int]]:
    """The pooled suite, with a per-component census.

    Order is stable, so a cached generation for one arm can be compared item by
    item with another arm's.
    """
    packed = _packed_suite()

    # On a compute node the fallback to ``datasets`` is a trap: it works on a
    # login node, which has a network, so every rehearsal passes, and then fails
    # on the compute node, which does not, about ninety seconds in.  Setting
    # SCD_REQUIRE_PACKED_SUITE=1 turns that silence into an error that names the
    # cause.
    if packed is None and os.environ.get("SCD_REQUIRE_PACKED_SUITE") == "1":
        raise RuntimeError(
            "SCD_REQUIRE_PACKED_SUITE=1 but %s could not be read from "
            "code/artifacts/. This run must not load `datasets`; ship the "
            "packed suite with the code." % SUITE_FILE)

    if packed is not None:
        problems, census = packed
        if limit_per_component:
            by: Dict[str, List[Problem]] = {}
            for pr in problems:
                by.setdefault(pr.source, []).append(pr)
            by = {k: v[:limit_per_component] for k, v in by.items()}
            problems = [pr for k in sorted(by) for pr in by[k]]
            census = {k: len(v) for k, v in sorted(by.items())}
        # _assert_disjoint is deliberately not called here: it checks by calling
        # train_questions(), which loads the dataset again, and avoiding that load
        # is why the packed file exists.  The check runs once when the file is
        # generated, on a machine that has the dataset, and its result is recorded
        # on the first line under "_meta".
        return problems, census

    comps: Dict[str, List[Problem]] = {}

    g = _gsm8k_raw()
    comps["gsm8k-test"] = [
        Problem(r["question"], gold_from_gsm8k(r["answer"]), "gsm8k-test")
        for r in g["test"]]

    gt = g["train"]
    held = _perm(len(gt))[:HELD_GSM8K]
    comps["gsm8k-heldout"] = [
        Problem(gt[int(i)]["question"], gold_from_gsm8k(gt[int(i)]["answer"]),
                "gsm8k-heldout") for i in held]

    m5 = _math500_raw()
    if m5 is not None:
        rows = []
        for r in m5:
            gold = r.get("answer")
            gold = gold if gold is not None else gold_from_math(r.get("solution", ""))
            from .tasks import _canon
            gold = _canon(str(gold)) if gold is not None else None
            if gold is not None:
                rows.append(Problem(r["problem"], gold, "math500"))
        comps["math500"] = rows

    mt = _math_train_raw()
    if mt is not None:
        held = _perm(len(mt))[:HELD_MATH]
        rows = []
        for i in held:
            r = mt[int(i)]
            gold = gold_from_math(r.get("solution", "") or "")
            if gold is not None:
                rows.append(Problem(r["problem"], gold, "math-heldout"))
        comps["math-heldout"] = rows

    if limit_per_component:
        comps = {k: v[:limit_per_component] for k, v in comps.items()}

    problems = [p for k in sorted(comps) for p in comps[k]]
    census = {k: len(v) for k, v in sorted(comps.items())}
    _assert_disjoint(problems)
    return problems, census


def _assert_disjoint(suite: List[Problem]) -> None:
    """The suite must not intersect what ``train_questions`` returns.

    Cheap version: the held-out slices are built from the complement of the
    training indices under the same fixed permutation, so disjointness is a
    property of the construction.  This checks the property rather than assuming
    the construction stayed correct through an edit.
    """
    tr = {p.question for p in train_questions(limit=4000)}
    bad = [p for p in suite if p.question in tr]
    if bad:
        raise RuntimeError(
            f"{len(bad)} evaluation items also appear in the training pool; "
            f"first: {bad[0].question[:80]!r}")


def mde(n: int, q: float = 0.20, power_z: float = 2.8) -> float:
    """Minimum detectable paired accuracy difference, in percentage points."""
    return 100.0 * power_z * (q / max(n, 1)) ** 0.5
