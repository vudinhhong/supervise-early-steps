r"""Turning corpus rows into training batches with a per-token loss mask.

The only subtle part is the step-to-token map.  ``scd.steps`` computes it once by
incremental tokenisation; nothing here is allowed to re-derive it with a different
rule, because a half-token disagreement between the mask used for training and the
mask used to define ``mu`` would make gate G3 measure nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from . import corpus, steps, tasks


@dataclass
class Batch:
    input_ids: torch.Tensor          # (1, L)
    targets: torch.Tensor            # (1, L-1)
    loss_mask: torch.Tensor          # (1, L-1) bool
    n_supervised: int
    key: str                         # question hash, for the teacher-logit cache
    resp_start: int


def build(tok, row: corpus.Row, max_len: int = 768) -> Optional[Batch]:
    head = tok(tasks.chat_prefix(tok, row.question),
               add_special_tokens=False)["input_ids"]
    body = "\n".join(row.steps)
    seg = steps.segment(tok, row.question, body, row.gold, row.source)
    resp = tok(body, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    ids = head + resp + [eos]
    if len(ids) > max_len:
        return None
    step_of = steps.step_of_token(seg, len(resp))
    mask_steps = row.step_mask or [True] * len(row.steps)
    keep = [bool(mask_steps[min(s, len(mask_steps) - 1)]) for s in step_of]
    keep.append(bool(mask_steps[-1]))                      # the eos token
    t = torch.tensor
    lm = torch.zeros(len(ids) - 1, dtype=torch.bool)
    lm[len(head) - 1:len(head) - 1 + len(keep)] = t(keep, dtype=torch.bool)
    return Batch(input_ids=t(ids).unsqueeze(0),
                 targets=t(ids[1:]).unsqueeze(0),
                 loss_mask=lm.unsqueeze(0),
                 n_supervised=int(lm.sum()),
                 key=cache_key(row), resp_start=len(head) - 1)


def cache_key(row: corpus.Row) -> str:
    import hashlib
    h = hashlib.sha1((row.question + "\x00" + "\n".join(row.steps)).encode()).hexdigest()
    return h[:24]


def collate(batches: Sequence[Batch], pad_id: int, device: str):
    """Right-pad for training (the loss mask already says what counts)."""
    L = max(b.input_ids.shape[1] for b in batches)
    n = len(batches)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    tgt = torch.full((n, L - 1), pad_id, dtype=torch.long)
    lm = torch.zeros((n, L - 1), dtype=torch.bool)
    for i, b in enumerate(batches):
        li = b.input_ids.shape[1]
        ids[i, :li] = b.input_ids[0]
        att[i, :li] = 1
        tgt[i, :li - 1] = b.targets[0]
        lm[i, :li - 1] = b.loss_mask[0]
    return (ids.to(device), att.to(device), tgt.to(device), lm.to(device))
