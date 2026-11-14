r"""Losses, all applied through an explicit position mask.

Two things drive this file.

*The mask is a first-class argument.*  Gate G3 manipulates which positions carry a
loss while holding the text, the objective and the number of supervised positions
fixed, so the mask cannot be a side effect of how the batch was built.  No loss
here normalises over anything but the mask.

*Nothing ever materialises a full-vocabulary float32 tensor.*  A first version
wrote the textbook form, ``F.log_softmax(logits.float(), -1)``, and it cost 8.7 GiB
of the 24 GiB allocation at batch 4, length 768 and a 152k vocabulary -- the
float32 copy, the log-softmax output, and their gradients, for a quantity of which
only 0.76 MiB is ever read.  It took the budget guard down on a 0.5B student with
adapters, where the arithmetic said 11 GiB.

The fix is exact, not an approximation.  Every objective below needs only two
things: the log-normaliser, and the student's logits at the teacher's top-$K$
columns.  The log-normaliser comes free from the cross-entropy that every arm
computes anyway, because

    CE(logits, target) = logsumexp(logits) - logits[target],

so ``logsumexp = CE + logits[target]``.  ``F.cross_entropy`` is a fused kernel that
accumulates the softmax in float32 internally without building a float32 copy of
its input.  From there every divergence is computed on ``(B, L, K)`` tensors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def _stats_chunk(lg: torch.Tensor, tgt: torch.Tensor):
    """Log-normaliser and cross-entropy for one slice, computed in float32."""
    f = lg.float()
    lse = torch.logsumexp(f, -1)
    return lse, lse - f.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


def token_stats(logits: torch.Tensor, targets: torch.Tensor,
                chunk: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(per-token cross-entropy, per-token log-normaliser)``, both float32.

    Chunked over positions and wrapped in activation checkpointing, which is what
    makes this both exact and small.  Each chunk's float32 slice exists only while
    that chunk is being reduced, and is recomputed rather than stored for backward,
    so the peak is one chunk instead of the whole ``(B, L, |V|)`` tensor.

    The obvious shortcut -- ``F.cross_entropy`` straight on the bfloat16 logits,
    then recovering the log-normaliser from ``CE + logits[target]`` -- is exact on
    paper and wrong in practice: that kernel returns a bfloat16-precision result,
    and the measured error was 0.046 nats. Forward KL reads the log-normaliser
    additively, so the error passes straight into the loss; the renormalised
    divergences happen to cancel a constant shift, which is exactly the kind of
    partial immunity that would have let the mistake through unnoticed in three
    arms out of five.
    """
    B, L, _ = logits.shape
    ces, lses = [], []
    for i in range(0, L, chunk):
        lg, tg = logits[:, i:i + chunk], targets[:, i:i + chunk]
        if torch.is_grad_enabled() and logits.requires_grad:
            lse_c, ce_c = torch.utils.checkpoint.checkpoint(
                _stats_chunk, lg, tg, use_reentrant=False)
        else:
            lse_c, ce_c = _stats_chunk(lg, tg)
        lses.append(lse_c)
        ces.append(ce_c)
    return torch.cat(ces, 1), torch.cat(lses, 1)


def logsumexp_full(logits: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """Log-normaliser without targets, chunked over positions.

    Only used where no cross-entropy is being computed -- the fidelity measurement
    in ``evaluation.kl_closure``, which runs under ``no_grad`` so the chunks are
    freed as it goes.
    """
    out = [torch.logsumexp(logits[:, i:i + chunk].float(), -1)
           for i in range(0, logits.shape[1], chunk)]
    return torch.cat(out, dim=1)


def _student_logprobs(logits: torch.Tensor, t_ids: torch.Tensor,
                      lse: torch.Tensor) -> torch.Tensor:
    """``log p_S`` at the teacher's support columns: ``(B, L, K)``, float32."""
    return logits.gather(-1, t_ids).float() - lse.unsqueeze(-1)


def _renormalised(slp: torch.Tensor) -> torch.Tensor:
    """``log p_S`` restricted to the support and renormalised over it."""
    return slp - torch.logsumexp(slp, -1, keepdim=True)


def ce(logits, targets, mask) -> torch.Tensor:
    return _masked_mean(token_stats(logits, targets)[0], mask)


def forward_kl(slp: torch.Tensor, t_logits: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    r"""``KL(p_T \| p_S)`` on the teacher's cached top-$K$ support.

    The support is the teacher's top-$K$, renormalised.  This is the standard
    approximation and it is biased: the set of teacher distributions consistent
    with a top-$K$ view has a total-variation diameter of 0.908 at $K=20$ on a
    realistic vocabulary.  The bias is identical for every arm, so it cannot
    explain a difference between arms -- which is the only claim these losses
    support.
    """
    tlp = F.log_softmax(t_logits.float(), -1)
    return _masked_mean((tlp.exp() * (tlp - slp)).sum(-1), mask)


def reverse_kl(slp, t_logits, mask) -> torch.Tensor:
    r"""``KL(p_S \| p_T)``, both restricted to the same support.

    Restricting a mode-seeking divergence to the teacher's top-$K$ is a real
    approximation -- the student's mass outside the support is invisible to it.
    Stated rather than hidden: the earlier round found reverse KL the strongest
    baseline, so it must not be quietly weakened.
    """
    tlp = F.log_softmax(t_logits.float(), -1)
    q = _renormalised(slp)
    return _masked_mean((q.exp() * (q - tlp)).sum(-1), mask)


def skew_kl(slp, t_logits, mask, alpha: float = 0.1) -> torch.Tensor:
    """``KL(p_T || (1-a) p_S + a p_T)`` -- bounded, and the usual modern default."""
    tlp = F.log_softmax(t_logits.float(), -1)
    q = _renormalised(slp)
    la = torch.log(torch.tensor(alpha, device=slp.device, dtype=slp.dtype))
    l1a = torch.log(torch.tensor(1.0 - alpha, device=slp.device, dtype=slp.dtype))
    mix = torch.logaddexp(q + l1a, tlp + la)
    return _masked_mean((tlp.exp() * (tlp - mix)).sum(-1), mask)


def jsd(slp, t_logits, mask) -> torch.Tensor:
    tlp = F.log_softmax(t_logits.float(), -1)
    q = _renormalised(slp)
    half = torch.log(torch.tensor(2.0, device=slp.device, dtype=slp.dtype))
    m = torch.logaddexp(q, tlp) - half
    return _masked_mean(0.5 * ((tlp.exp() * (tlp - m)).sum(-1)
                               + (q.exp() * (q - m)).sum(-1)), mask)


OBJECTIVES = ("ce", "fkl", "rkl", "skew", "jsd")


def distill_loss(name: str, logits, targets, t_ids, t_logits, mask,
                 lambda_ce: float = 1.0, alpha: float = 0.1):
    """``(total, parts)``.  Every KD arm keeps a cross-entropy term.

    Dropping CE from a KD arm is a different experiment: without it the student can
    stop producing well-formed text, and the comparison then measures fluency
    rather than the objective.  ``lambda_ce`` is identical across arms.
    """
    ce_tok, lse = token_stats(logits, targets)
    parts = {"ce": _masked_mean(ce_tok, mask)}
    if name == "ce" or t_ids is None:
        parts["kd"] = torch.zeros((), device=logits.device)
        return parts["ce"], parts
    slp = _student_logprobs(logits, t_ids, lse)
    if name == "fkl":
        parts["kd"] = forward_kl(slp, t_logits, mask)
    elif name == "rkl":
        parts["kd"] = reverse_kl(slp, t_logits, mask)
    elif name == "skew":
        parts["kd"] = skew_kl(slp, t_logits, mask, alpha)
    elif name == "jsd":
        parts["kd"] = jsd(slp, t_logits, mask)
    else:
        raise ValueError(f"unknown objective {name!r}; expected one of {OBJECTIVES}")
    return parts["kd"] + lambda_ce * parts["ce"], parts
