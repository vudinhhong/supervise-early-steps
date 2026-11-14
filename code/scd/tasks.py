"""Prompting and answer checking, identical for every arm.

The scoring protocol is fixed once here because the earlier evidential-lift
study showed how much it matters: a base student and a trained student fail to
state a final answer at very different rates, so "read the last number" rewards
a model that never finishes.  Every checkpoint gets the same forced cue and the
same extraction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

CUE = "\nThe answer is"

SYSTEM = ("You are a careful mathematician. Solve the problem step by step, "
          "one step per line, then state the final answer after 'The answer is'.")


@dataclass
class Problem:
    question: str
    gold: str
    source: str = ""


# Four worked examples in exactly the target format, used only to give a *base*
# model a fair chance to show what it knows.
#
# Why this exists.  With a non-instruction-tuned student the zero-shot score can be
# near zero because the model cannot produce the answer cue, not because it cannot
# do the arithmetic.  Measuring the gain against that baseline would let the study
# manufacture headroom out of format learning -- which is exactly the failure both
# surveys warn about, where imitation transfers style far more readily than
# substance.  So the reference point is the base model at its *best* available
# prompting, and the gain is measured against that.
#
# The examples are written here rather than sampled from a dataset so that they
# cannot overlap the evaluation suite.
FEWSHOT = [
    ("A box holds 12 pencils. Maria buys 4 boxes and gives away 7 pencils. "
     "How many pencils does she have left?",
     "She buys 4 * 12 = 48 pencils.\n"
     "After giving away 7 she has 48 - 7 = 41 pencils.\n"
     "The answer is 41"),
    ("A shirt costs 25 dollars and is discounted by 20 percent. "
     "What is the new price?",
     "The discount is 25 * 20 / 100 = 5 dollars.\n"
     "The new price is 25 - 5 = 20 dollars.\n"
     "The answer is 20"),
    ("Tom reads 18 pages a day for 5 days, then 12 pages a day for 3 days. "
     "How many pages in total?",
     "In the first stretch he reads 18 * 5 = 90 pages.\n"
     "In the second he reads 12 * 3 = 36 pages.\n"
     "Altogether that is 90 + 36 = 126 pages.\n"
     "The answer is 126"),
    ("If 3x + 7 = 25, what is x?",
     "Subtracting 7 from both sides gives 3x = 18.\n"
     "Dividing by 3 gives x = 6.\n"
     "The answer is 6"),
]


def chat_prefix(tok, question: str, shots: int = 0, plain: bool = False) -> str:
    """The prompt, identical for training and evaluation.

    ``shots`` and ``plain`` exist only for the base-model reference measurement.
    Every checkpoint that is compared with another is scored at the same settings.

    ``plain`` bypasses the chat template.  It is needed because the Qwen base
    checkpoints ship a chat template they were never trained on, so scoring a base
    model through it understates what the model knows -- and a baseline that is
    understated for a fixable reason turns format learning into apparent capability
    transfer, which is the confound the few-shot reference was added to close in
    the first place.  The reference is therefore the best of three promptings, not
    the best of two.
    """
    ex = FEWSHOT[:max(0, min(shots, len(FEWSHOT)))]
    if not plain:
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in ex:
            msgs += [{"role": "user", "content": q},
                     {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": question})
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            pass                           # tokenizer without a template
    body = "".join(f"Problem: {q}\nSolution:\n{a}\n\n" for q, a in ex)
    return f"{SYSTEM}\n\n{body}Problem: {question}\nSolution:\n"


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")

# A digit run longer than this is not an answer, it is a degenerate generation.
# No gold answer in the suite comes close; the largest are a few million.
MAX_ANSWER_DIGITS = 24


def _canon(s: str) -> str | None:
    """Canonical numeric string, or ``None`` if the text is not a usable number.

    Every rejection path here was reached by a real generation.  A base,
    non-instruction-tuned student that has not yet learned the answer format
    degenerates into long digit runs; ``float()`` turns those into ``inf`` and
    ``round(inf)`` raises ``OverflowError``, which took down an evaluation pass
    after 3072 of 3471 items.  Length is checked before ``float`` because the
    overflow is the *cheap* failure -- the expensive one is a silent ``inf``
    propagating into a bucket index somewhere later.
    """
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    if not s:
        return None
    if sum(c.isdigit() for c in s) > MAX_ANSWER_DIGITS:
        return None
    try:
        v = float(s)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(v):
        return None
    try:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
    except (OverflowError, ValueError):
        return None
    return f"{v:.6g}"


def extract_answer(text: str) -> str | None:
    """The number after the cue; the last number in the text if the cue is absent."""
    tail = text.rsplit(CUE, 1)[-1] if CUE in text else text
    nums = _NUM.findall(tail)
    if not nums and CUE in text:
        nums = _NUM.findall(text)
    return _canon(nums[-1]) if nums else None


def answer_matches(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    g = _canon(gold)
    return g is not None and pred == g


def gold_from_gsm8k(ans: str) -> str:
    return ans.split("####")[-1].strip()


_BOXED = re.compile(r"\\boxed\{")


def gold_from_math(solution: str) -> str | None:
    """Contents of the last \\boxed{...}, brace-matched.

    A regex cannot do this: ``\\boxed{\\frac{1}{2}}`` has nested braces.  Only
    answers that canonicalise to a number are kept, because the accuracy
    protocol above compares numbers; symbolic answers are dropped rather than
    scored wrongly.
    """
    m = None
    for m in _BOXED.finditer(solution):
        pass
    if m is None:
        return None
    i, depth = m.end(), 1
    while i < len(solution) and depth:
        depth += {"{": 1, "}": -1}.get(solution[i], 0)
        i += 1
    return _canon(solution[m.end():i - 1])
