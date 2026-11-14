r"""Shared plumbing for the GPU stages: corpus construction, caches, evaluation.

Every artefact is content-addressed and reused, which is what keeps the study
inside its budget: one teacher sampling pass serves every arm that shares that
teacher, and one top-K logit cache serves every objective.  It is also what makes
the comparisons exact -- two arms that differ only in a mask read byte-identical
text and byte-identical teacher logits.
"""

from __future__ import annotations

import json
import os
import time
import socket
import pathlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from scd import (corpus, coverage, data, evaluation, hf, runio, sampling,
                 teacherlogits, train as _train)
from scd.abstraction import N_STATES

ART = runio.ROOT / "artifacts"


def slug(name: str) -> str:
    return name.split("/")[-1].replace(".", "p")


def traces_path(teacher: str, n_prompts: int, n_samples: int, tag: str = "") -> pathlib.Path:
    return ART / f"traces_{slug(teacher)}_{n_prompts}x{n_samples}{tag}.jsonl"


def logits_dir(teacher: str, tag: str = "") -> pathlib.Path:
    return ART / f"logits_{slug(teacher)}{tag}"


def build_traces(teacher_name: str, n_prompts: int, n_samples: int,
                 max_new: int = 384, gen_batch: int = 16, seed: int = 0,
                 keep_correct_only: bool = True,
                 include_math: bool = True) -> List[corpus.Row]:
    """Sample teacher traces once, cache to disk, reuse forever."""
    path = traces_path(teacher_name, n_prompts, n_samples)
    if path.exists():
        rows = corpus.load(path)
        print(f"[common] reusing {len(rows)} traces from {path.name}")
        return rows
    probs = data.train_questions(limit=n_prompts, include_math=include_math)
    tok = hf.load_tokenizer(teacher_name)
    model = hf.load_model(teacher_name)
    print(f"[common] sampling {n_samples} traces for {len(probs)} prompts "
          f"from {teacher_name}  ({hf.mem()})")
    recs = sampling.sample_traces(model, tok, probs, n_samples, max_new=max_new,
                                  micro_batch=gen_batch, seed=seed)
    hf.free(model)
    rows = corpus.from_records(recs, origin="teacher")
    kept = [r for r in rows if r.correct] if keep_correct_only else rows
    corpus.mask_all(kept)
    corpus.save(kept, path)
    print(f"[common] kept {len(kept)}/{len(rows)} correct traces -> {path.name}")
    return kept


def build_logit_cache(teacher_name: str, rows: Sequence[corpus.Row],
                      topk: int = 64, micro_batch: int = 4,
                      tag: str = "") -> Tuple[teacherlogits.Cache, dict]:
    out = logits_dir(teacher_name, tag)
    tok = hf.load_tokenizer(teacher_name)
    model = hf.load_model(teacher_name)
    info = teacherlogits.build_cache(model, tok, rows, out, topk=topk,
                                     micro_batch=micro_batch)
    hf.free(model)
    print(f"[common] teacher logits: {info}")
    return teacherlogits.Cache(out), info


def student(student_name: str, lora_r: int = 32, seed: Optional[int] = None):
    """``lora_r = 0`` means full fine-tuning, in float32 (see hf.load_model).

    ``seed`` seeds the adapter initialisation.  It has to happen here and not in
    ``train.train``, which seeds only after the model already exists: PEFT draws
    ``lora_A`` from torch's global generator, and that generator is seeded from the
    operating system at process start.  Leaving it alone means every run gets a
    different adapter, so an arm is not reproducible from its seed and the spread
    over seeds mixes data order, which is controlled, with initialisation, which is
    not.  ``lora_B`` starts at zero, so the draw does not change the model at step
    zero, but it does set the direction the first gradients take.
    """
    # A merge must never train.  If an arm is not recognised as finished -- a
    # ledger key that drifted, a field renamed -- the stage would quietly build a
    # student and train it, which on a login node is hours that nobody expects.
    # Failing here, with the reason, is much cheaper.
    if os.environ.get("SCD_NO_TRAIN") == "1":
        raise RuntimeError(
            "SCD_NO_TRAIN=1 but a student %r was requested. During a merge every "
            "arm must already be in the ledger; an arm treated as unfinished "
            "means an identity key does not match, not that it needs retraining."
            % student_name)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    tok = hf.load_tokenizer(student_name)
    model = hf.load_model(student_name, train=True, full_finetune=(lora_r == 0))
    if lora_r > 0:
        model = hf.attach_lora(model, r=lora_r)
    else:
        for prm in model.parameters():
            prm.requires_grad_(True)
        n = sum(prm.numel() for prm in model.parameters())
        print(f"[common] full fine-tuning: {n/1e6:.0f}M trainable parameters")
    return model, tok


def coverage_of(rows: Sequence[corpus.Row],
                teacher_rows: Optional[Sequence[corpus.Row]] = None) -> dict:
    r"""``C2`` of the supervision measure against the teacher's occupancy.

    Estimator A only: it is the one that is paraphrase-invariant by construction,
    and gate G2 decides whether the hidden-state estimators may be used at all.
    """
    d = corpus.teacher_codes(teacher_rows if teacher_rows is not None else rows)
    mu = corpus.measure_codes(rows)
    out = coverage.c2_counting(d, mu, N_STATES)
    out["n_supervised_steps"] = sum(r.n_supervised_steps() for r in rows)
    out["n_rows"] = len(rows)
    return out


def suite(limit: int = 0):
    # A merge does not need the suite itself, only its length: every C.score call
    # lies on a path the merge skips, and `problems` reaches nothing but len() and
    # data.mde(len()).  Loading the dataset still happens, and it has caused three
    # separate failures: a missing cache in offline mode, a stall waiting on the
    # network from a login node, and a disk quota exhausted by the .lock files the
    # library writes even to read.
    #
    # So during a merge the suite is stubbed from the census recorded in the blob.
    # This path is taken only when SCD_NO_TRAIN=1, so a real run never reaches it.
    stub = os.environ.get("SCD_SUITE_CENSUS")
    if stub and os.environ.get("SCD_NO_TRAIN") == "1":
        census = json.loads(stub)
        n = sum(census.values())
        print(f"[common] evaluation suite n={n}  {census}  "
              f"MDE={data.mde(n):.2f} points  (stub for the merge; no dataset load)")
        return [None] * n, census
    problems, census = data.eval_suite(limit_per_component=limit)
    n = len(problems)
    print(f"[common] evaluation suite n={n}  {census}  "
          f"MDE={data.mde(n):.2f} points")
    return problems, census


def score(model, tok, problems, eval_batch: int = 16, max_new: int = 384,
          shots: int = 0, plain: bool = False) -> dict:
    return evaluation.accuracy(model, tok, problems, micro_batch=eval_batch,
                               max_new=max_new, shots=shots, plain=plain)


def append_run(tag: str, record: dict) -> None:
    """Append one arm's record to ``<results dir>/<tag>_runs.jsonl`` immediately.

    A stage that only writes at the end loses everything if it dies at arm 40 of
    100, and the main comparison is a 250 GPU-hour stage.  The JSONL is append-only
    and is the recovery path, not the primary artefact: the stage still writes its
    analysed blob at the end.
    """
    path = runio.results_dir() / f"{tag}_runs.jsonl"
    # Each record carries the revision that produced it.  Without that, the
    # resume path skips an arm already in the ledger whatever code wrote it, so
    # arms from a superseded revision would survive into the results beside arms
    # from the corrected one -- which is the thing a re-run exists to remove.
    record = dict(record)
    record.setdefault("code_sha", runio.code_sha())
    # Time and job id, on the record itself.
    #
    # Provenance otherwise has to be inferred from the blob and the logs, and both
    # can go missing: one stage left eighty records in a ledger dated a week after
    # the blob beside it, with no way left to say which job produced them.  A
    # record that carries its own provenance depends on no other file.
    record.setdefault("run_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    record.setdefault("pbs_jobid", os.environ.get("PBS_JOBID", ""))
    record.setdefault("host", socket.gethostname())
    with path.open("a") as f:
        f.write(json.dumps({k: v for k, v in record.items() if k != "per_item"},
                           default=float) + "\n")


def oom_safe(label: str, fn, *args, **kwargs):
    """Run one arm; on a CUDA out-of-memory, record it and carry on.

    One arm failing should cost one arm.  The alternative -- a stage-wide abort --
    is what a hand-set memory ceiling used to do, and it threw away a run whose
    hardware had two gigabytes spare.

    The failure is returned rather than swallowed, so the analysis can see which
    arms are missing and why instead of silently averaging over fewer seeds.

    Two causes, one list.  Whatever the caller records them in holds every arm that
    failed, and the message says which kind: a divergence is prefixed
    ``DIVERGED:``, an out-of-memory is not.  The guarantee that matters is the same
    either way -- the arm comes back as ``None`` and is never recorded as an
    accuracy.
    """
    try:
        return fn(*args, **kwargs), None
    except torch.cuda.OutOfMemoryError as e:        # noqa: F821 - torch imported below
        import torch as _t
        _t.cuda.empty_cache()
        msg = str(e).splitlines()[0]
        print(f"[oom] {label}: {msg}\n      arm skipped; the stage continues. "
              f"Lower GEN_BATCH / set GRAD_CHECKPOINTING=1 and rerun this arm.",
              flush=True)
        return None, msg
    except _train.Diverged as e:
        # A diverged arm used to come back as a finished run holding a destroyed
        # model, which scores near zero and is indistinguishable, downstream, from
        # a real collapse.  It is a failure, not a datum, and is reported as one.
        import torch as _t
        _t.cuda.empty_cache()
        msg = str(e).splitlines()[0]
        print(f"[diverged] {label}: {msg}\n      arm skipped; the stage continues.",
              flush=True)
        return None, "DIVERGED: " + msg


def shard(items: Sequence, nshards: int = 1, which: int = 0,
          reverse: bool = False) -> list:
    """Split an arm list across workers, the way G3 already did it.

    One arm per cluster job is what a scheduler wants: short walltime, restartable,
    and no two jobs writing the same file.  Splitting on position in a deterministic
    enumeration keeps the shards disjoint without the workers having to talk to each
    other, so each job can own its own results directory and the union is taken
    afterwards.  ``--list-arms`` prints the enumeration, so which index means which
    arm is on the record rather than implied by the loop order.
    """
    out = list(items)
    if reverse:
        out.reverse()
    if nshards > 1:
        out = [x for i, x in enumerate(out) if i % nshards == which]
    return out


def shard_args(ap) -> None:
    """The flags every training stage takes, so one job can run one arm."""
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the arms across this many workers")
    ap.add_argument("--shard", type=int, default=0, help="which shard to run")
    ap.add_argument("--list-arms", action="store_true",
                    help="print this shard's arms and exit, touching no GPU")
    ap.add_argument("--base-accuracy", type=float, default=None,
                    help="use this untrained accuracy instead of measuring it")


def probe_and_fit_rows(rows: Sequence[corpus.Row], frac: float = 0.05,
                       seed: int = 0):
    """A held-out slice of the arm's own measure (V0.7) and a fixed drift probe."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    k = max(8, int(frac * len(rows)))
    fit = [rows[int(i)] for i in idx[:k]]
    probe = [rows[int(i)] for i in idx[k:k + 256]]
    train = [rows[int(i)] for i in idx[k:]]
    return train, probe, fit
