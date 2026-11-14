r"""Scoring a checkpoint on the pooled suite, with per-item outcomes kept.

Per-item outcomes are kept because every comparison in the protocol is paired:
without them the only available test is over seeds, which answers a question about
this benchmark and not about the population.  Section 9 requires both.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from . import hf, sampling, tasks


@torch.no_grad()
def accuracy(model, tok, problems, max_new: int = 384, micro_batch: int = 16,
             greedy: bool = True, temperature: float = 0.8, n_samples: int = 1,
             progress_every: int = 512, shots: int = 0,
             plain: bool = False) -> Dict:
    was = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    prompts = [tasks.chat_prefix(tok, p.question, shots, plain)
               for p in problems]
    texts = sampling.generate(model, tok, prompts, n_samples=n_samples,
                              max_new=max_new, greedy=greedy,
                              temperature=temperature, micro_batch=micro_batch,
                              progress_every=progress_every)
    model.config.use_cache = was
    per_item, by_src = [], {}
    for k, t in enumerate(texts):
        p = problems[k // n_samples]
        ok = int(tasks.answer_matches(tasks.extract_answer(t), p.gold))
        per_item.append(ok)
        by_src.setdefault(p.source, []).append(ok)
    return {"accuracy": float(np.mean(per_item)), "n": len(per_item),
            "shots": shots, "plain": plain, "per_item": per_item,
            "by_source": {k: {"accuracy": float(np.mean(v)), "n": len(v)}
                          for k, v in sorted(by_src.items())},
            "mean_tokens": float(np.mean([len(tok(t)["input_ids"]) for t in texts]))}


@torch.no_grad()
def pass_at_k(model, tok, problems, ks=(1, 2, 4, 8, 16, 32, 64), n: int = 64,
              temperature: float = 0.8, max_new: int = 384,
              micro_batch: int = 16) -> Dict:
    r"""Unbiased pass@k from ``n`` samples, so one generation pass serves every k.

    The unbiased estimator is ``1 - C(n-c, k)/C(n, k)`` with ``c`` correct out of
    ``n``; the naive "fraction of problems with at least one correct among the
    first k" wastes samples and is biased upward at small k.
    """
    from math import comb
    prompts = [tasks.chat_prefix(tok, p.question) for p in problems]
    texts = sampling.generate(model, tok, prompts, n_samples=n, max_new=max_new,
                              greedy=False, temperature=temperature,
                              micro_batch=micro_batch)
    out = {}
    correct = []
    for i, p in enumerate(problems):
        c = sum(int(tasks.answer_matches(tasks.extract_answer(t), p.gold))
                for t in texts[i * n:(i + 1) * n])
        correct.append(c)
    for k in ks:
        if k > n:
            continue
        vals = [1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0)
                for c in correct]
        out[str(k)] = float(np.mean(vals))
    return {"pass_at_k": out, "n_samples": n, "n_problems": len(problems),
            "correct_counts": correct}


@torch.no_grad()
def kl_closure(model, tok, rows, cache, micro_batch: int = 4,
               max_rows: int = 512) -> float:
    r"""Mean ``KL(p_T || p_S)`` at the supervised positions, on the cached support.

    This is the fidelity number a distillation paper reports.  It is measured on
    the arm's *own* measure, deliberately: the whole claim under test is that this
    number can be small while capability is poor.
    """
    from . import batching, losses
    dev = hf.device()
    tot, n = 0.0, 0
    buf = []
    for r in rows[:max_rows]:
        b = batching.build(tok, r)
        if b is None or b.n_supervised == 0:
            continue
        buf.append(b)
        if len(buf) < micro_batch:
            continue
        ids, att, tgt, lm = batching.collate(buf, tok.pad_token_id, dev)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
        t_ids, t_lg = cache.stack(buf, logits.shape[1], dev)
        if t_ids is not None:
            lse = losses.logsumexp_full(logits)
            slp = losses._student_logprobs(logits, t_ids, lse)
            tot += float(losses.forward_kl(slp, t_lg, lm)); n += 1
        del logits
        buf = []
    return tot / n if n else float("nan")
