r"""Generation, prefix continuation, and hidden-state readout.

Three operations, each written to keep one model resident at a time within a
30 GB ceiling.

``sample`` draws continuations for a list of prompts.

``continue_from_prefix`` is the operation the method needs and off-the-shelf
distillation code does not have: given the first ``t`` steps of a trace, ask the
teacher to carry on.  That is how supervision gains coverage at a state the
supervision did not previously reach.  It is *not* on-policy distillation --
nothing scores the student's tokens; the teacher simply writes from a state it
would not otherwise have visited.

``hidden_at_prefixes`` returns the student's last-layer state at every reasoning-
step boundary of a trace.  One forward per trace suffices for all of them
because attention is causal -- an obvious point that is easy to miss and worth
a factor of six in cost.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from . import hf, steps, tasks


def _pad_batch(tok, id_lists: Sequence[Sequence[int]], device: str):
    n = max(len(x) for x in id_lists)
    pad = tok.pad_token_id
    ids = torch.full((len(id_lists), n), pad, dtype=torch.long)
    att = torch.zeros((len(id_lists), n), dtype=torch.long)
    for i, x in enumerate(id_lists):           # left padding
        ids[i, n - len(x):] = torch.tensor(x, dtype=torch.long)
        att[i, n - len(x):] = 1
    return ids.to(device), att.to(device)


@torch.no_grad()
def generate(model, tok, prompts: Sequence[str], n_samples: int = 1,
             max_new: int = 320, temperature: float = 0.8, greedy: bool = False,
             micro_batch: int = 16, seed: int = 0,
             progress_every: int = 0) -> List[str]:
    """Continuations in the same order as ``prompts``, repeated ``n_samples`` times.

    Rows are sorted by length inside a micro-batch so padding does not dominate,
    then restored.  ``micro_batch`` is the only VRAM knob that matters here; the
    caller lowers it, never the dtype.
    """
    dev = hf.device()
    rows = [p for p in prompts for _ in range(n_samples)]
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in rows]
    order = np.argsort([len(e) for e in enc])
    out: List[str] = [""] * len(rows)
    eos = sorted(hf.stop_ids(tok))
    torch.manual_seed(seed)
    done = 0
    for s in range(0, len(order), micro_batch):
        sel = order[s:s + micro_batch]
        ids, att = _pad_batch(tok, [enc[i] for i in sel], dev)
        gen = model.generate(
            input_ids=ids, attention_mask=att, max_new_tokens=max_new,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else 0.95,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos or tok.eos_token_id,
        )
        new = gen[:, ids.shape[1]:]
        for j, i in enumerate(sel):
            out[int(i)] = tok.decode(new[j], skip_special_tokens=True)
        done += len(sel)
        if progress_every and done % progress_every < micro_batch:
            print(f"    [gen] {done}/{len(rows)}  {hf.mem()}", flush=True)
        hf.check_budget("generate")
    return out


def sample_traces(model, tok, problems, n_samples: int, max_new: int = 320,
                  temperature: float = 0.8, micro_batch: int = 16,
                  seed: int = 0, progress_every: int = 256) -> List[dict]:
    """Traces with the cue appended, segmented, and verified."""
    prompts = [tasks.chat_prefix(tok, p.question) for p in problems]
    texts = generate(model, tok, prompts, n_samples=n_samples, max_new=max_new,
                     temperature=temperature, micro_batch=micro_batch, seed=seed,
                     progress_every=progress_every)
    recs = []
    for k, t in enumerate(texts):
        p = problems[k // n_samples]
        seg = steps.segment(tok, p.question, t.strip(), p.gold, p.source)
        recs.append({"question": p.question, "gold": p.gold, "source": p.source,
                     "response": seg.response, "steps": seg.steps,
                     "answer": seg.answer, "correct": bool(seg.correct),
                     "sample": k % n_samples})
    return recs


@torch.no_grad()
def continue_from_prefix(model, tok, items: Sequence[dict], max_new: int = 320,
                         temperature: float = 0.8, micro_batch: int = 16,
                         seed: int = 0, progress_every: int = 256) -> List[dict]:
    """``items`` are ``{question, gold, prefix_steps}``; the teacher writes on.

    The prefix is fed as the *beginning of the assistant turn*, not as part of
    the user message, so the continuation is drawn from the teacher's own
    conditional at that state rather than from its reaction to being shown work.
    That distinction is the whole point: mis-implementing it turns a coverage
    intervention into a prompting trick.
    """
    prompts = []
    for it in items:
        head = tasks.chat_prefix(tok, it["question"])
        pre = "\n".join(it["prefix_steps"])
        prompts.append(head + (pre + "\n" if pre else ""))
    texts = generate(model, tok, prompts, n_samples=1, max_new=max_new,
                     temperature=temperature, micro_batch=micro_batch, seed=seed,
                     progress_every=progress_every)
    out = []
    for it, t in zip(items, texts):
        body = ("\n".join(it["prefix_steps"]) + "\n" + t.strip()).strip() \
            if it["prefix_steps"] else t.strip()
        seg = steps.segment(tok, it["question"], body, it["gold"],
                            it.get("source", ""))
        out.append({"question": it["question"], "gold": it["gold"],
                    "source": it.get("source", ""), "response": seg.response,
                    "steps": seg.steps, "answer": seg.answer,
                    "correct": bool(seg.correct),
                    "branch_at": len(it["prefix_steps"]),
                    "parent": it.get("parent", -1)})
    return out


@torch.no_grad()
def hidden_at_prefixes(model, tok, recs: Sequence[dict], layer: int = -1,
                       micro_batch: int = 8, max_len: int = 768,
                       progress_every: int = 512) -> List[np.ndarray]:
    """For each record, an ``(m+1, hidden)`` array: one vector per step boundary.

    Boundary ``t`` is read at the last token of the first ``t`` steps, which for
    ``t = 0`` is the last prompt token.  Because attention is causal that vector
    does not depend on anything after it, so a single forward over the full trace
    yields every boundary of that trace.
    """
    dev = hf.device()
    out: List[np.ndarray] = []
    buf_ids, buf_pos, buf_i = [], [], []

    def flush():
        if not buf_ids:
            return
        ids, att = _pad_batch(tok, buf_ids, dev)
        h = model(input_ids=ids, attention_mask=att,
                  output_hidden_states=True).hidden_states[layer]
        n = ids.shape[1]
        for j, positions in enumerate(buf_pos):
            off = n - len(buf_ids[j])          # left padding offset
            idx = [min(off + p, n - 1) for p in positions]
            out[buf_i[j]] = h[j, idx, :].float().cpu().numpy()
        buf_ids.clear(); buf_pos.clear(); buf_i.clear()
        hf.check_budget("hidden")

    for r in recs:
        head = tok(tasks.chat_prefix(tok, r["question"]),
                   add_special_tokens=False)["input_ids"]
        positions, ids = [len(head) - 1], list(head)
        for t, st in enumerate(r["steps"]):
            piece = tok(("\n" if t else "") + st,
                        add_special_tokens=False)["input_ids"]
            ids += piece
            positions.append(len(ids) - 1)
        if len(ids) > max_len:                  # keep the head, drop the tail
            keep = [p for p in positions if p < max_len]
            ids, positions = ids[:max_len], keep or [max_len - 1]
        out.append(np.zeros((len(positions), 1), dtype=np.float32))
        buf_ids.append(ids); buf_pos.append(positions); buf_i.append(len(out) - 1)
        if len(buf_ids) >= micro_batch:
            flush()
            if progress_every and len(out) % progress_every < micro_batch:
                print(f"    [hidden] {len(out)}/{len(recs)}  {hf.mem()}", flush=True)
    flush()
    return out


@torch.no_grad()
def entropy_at_prefixes(model, tok, recs, micro_batch: int = 8,
                        max_len: int = 768, chunk: int = 64,
                        progress_every: int = 2048):
    r"""Predictive entropy of ``model`` at every reasoning-step boundary.

    The student-side selection signal that design doc §8.1 registered as
    alternative (c) for gate G6 and that nothing had implemented -- ``alt_select``
    advertised it in its docstring and raised ``ValueError`` when asked for it.
    Writing it now exhausts the registered options rather than fishing for a new
    one: the theory-derived rule has already failed cleanly across four budgets.

    One forward per trace serves every boundary of that trace, as with the hidden
    states, because attention is causal.  The softmax is reduced in float32 slices
    for the reason given in ``scd.losses``: a full-vocabulary float32 tensor at
    this batch and length is gigabytes for a quantity of which one number per
    position is read.
    """
    dev = hf.device()
    out = []
    buf_ids, buf_pos, buf_i = [], [], []

    def flush():
        if not buf_ids:
            return
        ids, att = _pad_batch(tok, buf_ids, dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        n = ids.shape[1]
        ent = torch.empty(logits.shape[:2], dtype=torch.float32, device=dev)
        for i in range(0, n, chunk):
            sl = logits[:, i:i + chunk].float()
            lp = torch.log_softmax(sl, -1)
            ent[:, i:i + chunk] = -(lp.exp() * lp).sum(-1)
            del sl, lp
        for j, positions in enumerate(buf_pos):
            off = n - len(buf_ids[j])
            idx = [min(off + q, n - 1) for q in positions]
            out[buf_i[j]] = ent[j, idx].cpu().numpy()
        del logits, ent
        buf_ids.clear(); buf_pos.clear(); buf_i.clear()
        hf.check_budget("entropy")

    for r in recs:
        head = tok(tasks.chat_prefix(tok, r["question"] if isinstance(r, dict)
                                     else r.question), add_special_tokens=False)["input_ids"]
        steps = r["steps"] if isinstance(r, dict) else r.steps
        positions, ids = [len(head) - 1], list(head)
        for k, st in enumerate(steps):
            piece = tok(("\n" if k else "") + st, add_special_tokens=False)["input_ids"]
            ids += piece
            positions.append(len(ids) - 1)
        if len(ids) > max_len:
            keep = [q for q in positions if q < max_len]
            ids, positions = ids[:max_len], keep or [max_len - 1]
        out.append(np.zeros(len(positions), dtype=np.float32))
        buf_ids.append(ids); buf_pos.append(positions); buf_i.append(len(out) - 1)
        if len(buf_ids) >= micro_batch:
            flush()
            if progress_every and len(out) % progress_every < micro_batch:
                print(f"    [entropy] {len(out)}/{len(recs)}  {hf.mem()}", flush=True)
    flush()
    return out
