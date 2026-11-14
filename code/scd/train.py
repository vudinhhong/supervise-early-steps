r"""One trainer, used by every arm.

The invariants below are the ones the previous round violated, so they are enforced
here rather than remembered:

*Every arm reports its parameter displacement* (V0.3).  An arm that moved a sixth
as far as another has not been compared to it -- that is how a tangent-projected
objective came to look like a winner.

  Read that column for what it can tell you.  Under AdamW the update magnitude is
  normalised by the gradient's own second moment, so the displacement is set
  mostly by the learning rate and the step count: measured on a fixed task it
  scales linearly in the learning rate (1.26e-4 at lr 1e-4, 1.26e-3 at lr 1e-3)
  and by roughly the square root of the step count, while varying only about 30%
  across the five objectives at matched learning rate and steps.  It is therefore
  a detector of schedule mismatches -- precisely the failure it was added for --
  and not a measure of how differently two objectives move the model.  For the
  latter, read ``probe_drift``, the change in mean log-probability on a fixed
  probe set, which is a functional rather than a parametric distance.

*Every arm reports its fit on its own supervision* (V0.7).  That is the only
legitimate exclusion rule: a run that failed to fit the measure it was trained on
had an optimisation failure, and dropping it is a decision about the independent
variable.  Dropping a run because its accuracy came out low is selection on the
outcome, and the theory predicts precisely that some measures produce low accuracy.

*Learning rate is swept per arm family* (V0.4), not frozen from the baseline arm.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import batching, corpus, hf, losses, teacherlogits



class Diverged(RuntimeError):
    """Training produced a non-finite loss, so the arm yielded no measurement.

    Kept distinct from an out-of-memory failure: one says the hardware was too
    small, the other says the optimisation came apart.  Both must be reported,
    and neither may be recorded as an accuracy.
    """

@dataclass
class TrainConfig:
    objective: str = "ce"          # ce | fkl | rkl | skew | jsd
    lr: float = 1e-4
    epochs: int = 3
    batch: int = 4
    accum: int = 4
    lora_r: int = 32
    lambda_ce: float = 1.0
    alpha: float = 0.1
    max_len: int = 768
    seed: int = 0
    warmup: float = 0.03
    grad_clip: float = 1.0
    log_every: int = 25
    # Read from the environment so a single knob in run.sh reaches every stage
    # without threading an argument through six experiment scripts.
    grad_checkpointing: bool = field(
        default_factory=lambda: bool(int(os.environ.get("SCD_GRAD_CHECKPOINTING", "0"))))


def _probe_logprobs(model, tok, probe_rows, device, limit: int = 256):
    """Mean log-probability on a fixed probe set -- the drift reference (V0.3)."""
    out = []
    with torch.no_grad():
        for r in probe_rows[:limit]:
            b = batching.build(tok, r)
            if b is None:
                continue
            ids = b.input_ids.to(device)
            lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
            out.append(float(lp.gather(-1, b.targets.to(device)[0].unsqueeze(-1)).mean()))
    return float(np.mean(out)) if out else float("nan")


def _held_out_fit(model, tok, rows, device, cfg, cache) -> float:
    """Loss on positions drawn from the arm's *own* measure (V0.7)."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for r in rows:
            b = batching.build(tok, r, cfg.max_len)
            if b is None or b.n_supervised == 0:
                continue
            ids, att, tgt, lm = batching.collate([b], tok.pad_token_id, device)
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
            t_ids = t_lg = None
            if cfg.objective != "ce" and cache is not None:
                t_ids, t_lg = cache.stack([b], logits.shape[1], device)
            loss, _ = losses.distill_loss(cfg.objective, logits, tgt, t_ids, t_lg,
                                          lm, cfg.lambda_ce, cfg.alpha)
            tot += float(loss); n += 1
    return tot / n if n else float("nan")


def train(model, tok, rows: Sequence[corpus.Row], cfg: TrainConfig,
          cache: Optional[teacherlogits.Cache] = None,
          probe_rows: Sequence[corpus.Row] = (),
          fit_rows: Sequence[corpus.Row] = ()) -> Dict:
    dev = hf.device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # One large model resident at a time (see hf.assert_sole_resident).  Checked
    # here rather than trusted: the leak that made this necessary reached the
    # server twice, and both times it surfaced as a budget failure twenty minutes
    # into training rather than as the missing release it actually was.
    hf.assert_sole_resident(model, "start of training", slack_gib=4.0)

    built = [b for b in (batching.build(tok, r, cfg.max_len) for r in rows)
             if b is not None and b.n_supervised > 0]
    if not built:
        raise RuntimeError("no trainable rows: every row was too long or fully masked")
    n_sup = sum(b.n_supervised for b in built)

    theta0 = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}
    probe0 = _probe_logprobs(model, tok, probe_rows, dev) if probe_rows else float("nan")
    fit0 = _held_out_fit(model, tok, fit_rows, dev, cfg, cache) if fit_rows else float("nan")

    if cfg.grad_checkpointing:
        # Trades about a third of the step time for roughly a tenth of the
        # activation memory.  Off by default because it changes wall clock, which
        # is reported; on when the budget guard says the activations are binding.
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(built) / (cfg.batch * cfg.accum))
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(total, 1),
        pct_start=cfg.warmup, anneal_strategy="cos")

    model.train()
    hist, step, t0 = [], 0, time.time()
    # The last few finite losses.  Without them a divergence report says only
    # "nan at step 18", which does not separate a loss that grew until it broke
    # from one bad batch in an otherwise normal sequence, and the two call for
    # different responses.
    recent: list = []
    rng = np.random.default_rng(cfg.seed)
    for ep in range(cfg.epochs):
        order = rng.permutation(len(built))
        opt.zero_grad(set_to_none=True)
        for j in range(0, len(order), cfg.batch):
            mb = [built[int(i)] for i in order[j:j + cfg.batch]]
            ids, att, tgt, lm = batching.collate(mb, tok.pad_token_id, dev)
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
            t_ids = t_lg = None
            if cfg.objective != "ce" and cache is not None:
                t_ids, t_lg = cache.stack(mb, logits.shape[1], dev)
                if t_ids is None:
                    raise RuntimeError("teacher-logit cache incomplete for this arm; "
                                       "run the `logits` stage first")
            loss, parts = losses.distill_loss(cfg.objective, logits, tgt, t_ids,
                                              t_lg, lm, cfg.lambda_ce, cfg.alpha)
            # A non-finite loss destroys the weights on the next step, and nothing
            # downstream can tell the wreck apart from a genuinely bad arm: the run
            # finishes, scores near zero, and gets averaged in as if it were a
            # measurement.  ``clip_grad_norm_`` does not help, because it propagates
            # NaN rather than rejecting it.  So stop here and say why.
            if not torch.isfinite(loss):
                raise Diverged(
                    "loss became %s at epoch %d, optimizer step %d "
                    "(objective=%s lr=%g lr_now=%g seed=%d); last finite losses "
                    "%s; the arm is not reported as a measurement"
                    % (float(loss.detach()), ep, step, cfg.objective, cfg.lr,
                       float(sched.get_last_lr()[0]), cfg.seed,
                       ["%d:%.4f" % r for r in recent[-6:]]))
            recent.append((step, float(loss.detach())))
            # One-shot breakdown on the first micro-batch.  The first time the
            # budget guard fired, the reply was a guess: the arithmetic said
            # 11 GiB and the guard said 23.9. Printing where it goes costs one
            # line in the log and removes the guessing.
            if step == 0 and ep == 0 and j == 0 and torch.cuda.is_available():
                g = 2 ** 30
                print(f"    [mem] after forward+loss: "
                      f"live {torch.cuda.memory_allocated()/g:.2f} GiB, "
                      f"peak {torch.cuda.max_memory_allocated()/g:.2f} GiB, "
                      f"logits {logits.numel()*logits.element_size()/g:.2f} GiB "
                      f"({tuple(logits.shape)}, {logits.dtype})", flush=True)
            del logits                       # the biggest live tensor; drop the
            (loss / cfg.accum).backward()    # Python reference before backward
            if step == 0 and ep == 0 and j == 0 and torch.cuda.is_available():
                print(f"    [mem] after backward:     "
                      f"live {torch.cuda.memory_allocated()/2**30:.2f} GiB, "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
                      flush=True)
            if (j // cfg.batch + 1) % cfg.accum == 0:
                # clip_grad_norm_ returns the gradient norm *before* clipping,
                # and an earlier version discarded it.  That was the real defect:
                # when the norm is inf or nan the clip multiplies every gradient
                # by nan, opt.step() writes nan into the weights, and the loss
                # check only notices on the following step, by which time the
                # model is wrecked.  Four arms in one run broke this way, all of
                # them at the same optimizer step.
                #
                # Checking here stops before the weights are destroyed and names
                # the cause: a broken gradient, not a broken loss.
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
                if not torch.isfinite(gn):
                    raise Diverged(
                        "gradient norm became %s at epoch %d, optimizer step %d "
                        "(objective=%s lr=%g lr_now=%g seed=%d); last finite "
                        "losses %s; the weights were NOT updated, so this is a "
                        "broken gradient and not a model already wrecked"
                        % (float(gn), ep, step, cfg.objective, cfg.lr,
                           float(sched.get_last_lr()[0]), cfg.seed,
                           ["%d:%.4f" % r for r in recent[-6:]]))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0:
                    hist.append({"epoch": ep, "step": step,
                                 "loss": float(loss), "ce": float(parts["ce"]),
                                 "kd": float(parts["kd"]),
                                 "lr": float(sched.get_last_lr()[0]),
                                 "seconds": time.time() - t0})
                    print(f"    [train] ep{ep} step{step}/{total} "
                          f"loss={float(loss):.4f} ce={float(parts['ce']):.4f} "
                          f"kd={float(parts['kd']):.4f}  {hf.mem()}", flush=True)
            hf.check_budget("train")

    model.eval()
    disp = 0.0
    for n, p in model.named_parameters():
        if n in theta0:
            disp += float(((p.detach() - theta0[n]) ** 2).sum())
    n_param = sum(p.numel() for n, p in model.named_parameters() if n in theta0)
    probe1 = _probe_logprobs(model, tok, probe_rows, dev) if probe_rows else float("nan")
    fit1 = _held_out_fit(model, tok, fit_rows, dev, cfg, cache) if fit_rows else float("nan")

    return {"history": hist, "steps": step, "optimizer_steps_planned": total,
            "n_rows": len(built), "n_supervised_positions": n_sup,
            "displacement": math.sqrt(disp) / math.sqrt(max(n_param, 1)),
            "displacement_raw": math.sqrt(disp), "n_trainable": n_param,
            "probe_logprob_before": probe0, "probe_logprob_after": probe1,
            "probe_drift": probe1 - probe0 if probe_rows else float("nan"),
            "own_supervision_loss_before": fit0,
            "own_supervision_loss_after": fit1,
            "own_supervision_loss_drop":
                (fit0 - fit1) / abs(fit0) if fit_rows and fit0 not in (0.0,) else float("nan"),
            "train_seconds": time.time() - t0, "config": cfg.__dict__}
