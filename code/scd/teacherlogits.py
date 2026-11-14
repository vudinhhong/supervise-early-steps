r"""Top-K teacher logits, cached at the positions that can ever carry a loss.

Cost control, not an optimisation detail: caching the full vocabulary at every
position of a 40k-trace corpus is tens of gigabytes, and caching the top 64 at the
supervised positions is under one.  The cache is keyed by a hash of (question,
trace) so two arms that share text share the cache -- which is also what makes the
gate G3 comparison exactly matched on the teacher side.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Sequence

import numpy as np
import torch

from . import batching, corpus, hf


@torch.no_grad()
def build_cache(teacher, tok, rows: Sequence[corpus.Row], out: pathlib.Path,
                topk: int = 64, micro_batch: int = 4, max_len: int = 768,
                progress_every: int = 500) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    dev = hf.device()
    made, skipped = 0, 0
    buf: List[batching.Batch] = []

    def flush():
        nonlocal made
        if not buf:
            return
        ids, att, _, _ = batching.collate(buf, tok.pad_token_id, dev)
        logits = teacher(input_ids=ids, attention_mask=att).logits[:, :-1]
        v, i = torch.topk(logits.float(), k=topk, dim=-1)
        for j, b in enumerate(buf):
            L = b.input_ids.shape[1] - 1
            np.savez_compressed(out / f"{b.key}.npz",
                                ids=i[j, :L].cpu().numpy().astype(np.int32),
                                logits=v[j, :L].cpu().numpy().astype(np.float16))
            made += 1
        buf.clear()
        hf.check_budget("teacher-logits")

    for r in rows:
        b = batching.build(tok, r, max_len=max_len)
        if b is None:
            skipped += 1
            continue
        if (out / f"{b.key}.npz").exists():
            skipped += 1
            continue
        buf.append(b)
        if len(buf) >= micro_batch:
            flush()
            if progress_every and made % progress_every < micro_batch:
                print(f"    [logits] {made} cached  {hf.mem()}", flush=True)
    flush()
    return {"cached": made, "skipped": skipped, "topk": topk, "dir": str(out)}


class Cache:
    """Read the logit cache: a directory of .npz files, or one .zip holding them.

    The bundle exists because of file counts, not bytes.  The 6000x4 corpus is
    15,994 .npz files; on a filesystem with a quota on the number of files that is
    a noticeable share of it, while 864 MB of space is not.  Collapsing them into
    one file keeps lookup by key, since a .npz is already a zip and zipfile reads
    members individually.
    """

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self._zip = None
        if self.path.is_file() and self.path.suffix == ".zip":
            import zipfile
            self._zip = zipfile.ZipFile(self.path)
            self._names = set(self._zip.namelist())

    def get(self, key: str, length: int, device: str):
        if self._zip is not None:
            name = f"{key}.npz"
            if name not in self._names:
                return None, None
            import io as _io
            z = np.load(_io.BytesIO(self._zip.read(name)))
        else:
            f = self.path / f"{key}.npz"
            if not f.exists():
                return None, None
            z = np.load(f)
        ids = torch.from_numpy(z["ids"][:length].astype(np.int64)).to(device)
        lg = torch.from_numpy(z["logits"][:length].astype(np.float32)).to(device)
        return ids, lg

    def stack(self, batches, length: int, device: str):
        """``(B, L, K)`` tensors aligned with ``collate``; ``None`` if any is missing."""
        out_i, out_l = [], []
        for b in batches:
            i, l = self.get(b.key, b.input_ids.shape[1] - 1, device)
            if i is None:
                return None, None
            pad = length - i.shape[0]
            if pad > 0:
                i = torch.cat([i, i[-1:].expand(pad, -1)], 0)
                l = torch.cat([l, torch.zeros(pad, l.shape[1], device=device)], 0)
            out_i.append(i[:length]); out_l.append(l[:length])
        return torch.stack(out_i), torch.stack(out_l)
