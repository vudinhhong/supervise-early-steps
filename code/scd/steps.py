"""Step segmentation and prefixes -- the objects the whole study is defined on.

A *state* is a prefix at a reasoning-step boundary, not at an arbitrary token.
Section 1.1 of the protocol argues the choice: a trace has of order 250 token
positions and of order 6 step boundaries, and most token positions are function
words that no reasoning step depends on.  Including them dilutes every measure
taken here.

The mapping from steps to token positions is what the loss masks need, so it is
computed once, exactly, by tokenising incrementally.  Nothing downstream is
allowed to re-derive it with a different rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import tasks


def split_steps(response: str) -> List[str]:
    """Non-empty lines, cue line kept as the final step."""
    parts = [s for s in response.split("\n") if s.strip()]
    return parts if parts else [response]


@dataclass
class Segmented:
    question: str
    response: str
    steps: List[str]
    prefixes: List[str]          # prefixes[t] = first t steps joined, t = 0..m
    tok_end: List[int]           # tok_end[t] = #response tokens in prefixes[t]
    gold: str = ""
    answer: str | None = None
    correct: bool = False
    source: str = ""

    @property
    def m(self) -> int:
        return len(self.steps)


def segment(tok, question: str, response: str, gold: str = "",
            source: str = "") -> Segmented:
    steps = split_steps(response)
    prefixes, tok_end = [""], [0]
    for t in range(len(steps)):
        pre = "\n".join(steps[:t + 1])
        prefixes.append(pre)
        tok_end.append(len(tok(pre, add_special_tokens=False)["input_ids"]))
    ans = tasks.extract_answer(response)
    return Segmented(question=question, response=response, steps=steps,
                     prefixes=prefixes, tok_end=tok_end, gold=gold,
                     answer=ans, correct=tasks.answer_matches(ans, gold),
                     source=source)


def step_of_token(seg: Segmented, n_resp_tokens: int) -> List[int]:
    """Step index for each response token position, length ``n_resp_tokens``.

    ``tok_end`` is monotone by construction, so this is a search over at most
    ``m`` boundaries per position.  Positions past the last boundary (the
    trailing end-of-turn token) are attributed to the last step.
    """
    out, t = [], 0
    for i in range(n_resp_tokens):
        while t + 1 < len(seg.tok_end) and i >= seg.tok_end[t + 1]:
            t += 1
        out.append(min(t, seg.m - 1))
    return out
