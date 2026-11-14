"""Model loading and VRAM accounting under a stated budget.

Three rules the rest of the code relies on.

*One large model resident at a time.*  A 7B teacher in bfloat16 is 14.2 GB and a
1.5B student with adapters plus optimiser state is about 5 GB; both fit, but not
with a generation cache and training activations on top.  Every stage therefore
loads the teacher, writes what it needs to disk, frees it, and only then loads the
student.  ``free`` and ``assert_sole_resident`` exist to make that sequencing
checkable rather than assumed -- a leak here cost two server runs.

*bfloat16 for inference, and for training weights unless the whole model is
trainable.*  The coverage estimators difference nothing, so half precision is safe;
full fine-tuning switches to float32 because AdamW on bfloat16 weights with no
float32 master copy loses small updates to rounding.

*Left padding everywhere.*  Both generation and the hidden-state readout need the
last real token of each row to sit at the same index.
"""

from __future__ import annotations

import gc
import os

import torch

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# A stated budget, used for reporting and for one advisory warning.  It is not a
# limit: see check_budget for why making it one was a mistake.
VRAM_BUDGET_GB = float(os.environ.get("SCD_VRAM_GB", "30"))
HARD_LIMIT = bool(int(os.environ.get("SCD_VRAM_HARD", "0")))
_warned: set = set()
# Sticky high-water mark.  ``free`` resets torch's counter so that each stage's
# peak is attributable, which means the counter is near zero by the time a result
# blob is written -- the first run reported "peak allocated 0.0 GiB" after touching
# 9.9. This remembers the largest value seen across the whole process.
_peak_seen_bytes: int = 0


def _remember_peak() -> None:
    global _peak_seen_bytes
    if torch.cuda.is_available():
        _peak_seen_bytes = max(_peak_seen_bytes,
                               torch.cuda.max_memory_allocated())


# --------------------------------------------------------------------- loading


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def tokenizer_fingerprint(name: str) -> str:
    """Hash of the vocabulary.

    Validity condition V5.9: forward KL between two models is undefined unless they
    share a tokenizer, and a pair that differs must not enter the G4 grid.
    Comparing ``vocab_size`` is not enough -- two Qwen generations can agree on the
    size and disagree on a handful of added tokens.
    """
    import hashlib
    v = load_tokenizer(name).get_vocab()
    h = hashlib.sha256()
    for tokstr, i in sorted(v.items(), key=lambda kv: kv[1]):
        h.update(f"{i}\t{tokstr}\n".encode("utf-8", "surrogatepass"))
    return h.hexdigest()[:16]


WAIT_FOR_VRAM = bool(int(os.environ.get("SCD_WAIT_FOR_VRAM", "1")))
MIN_FREE_GIB = float(os.environ.get("SCD_MIN_FREE_GIB", "16"))
WAIT_TIMEOUT_MIN = float(os.environ.get("SCD_VRAM_WAIT_MIN", "90"))


def free_vram_gib() -> float:
    if not torch.cuda.is_available():
        return float("inf")
    return torch.cuda.mem_get_info()[0] / 2 ** 30


def wait_for_vram(min_free_gib: float | None = None, tag: str = "") -> None:
    """Block until the device has room, rather than dying when a neighbour grows.

    This machine is shared.  A run of G6 was killed mid-load when the job on the
    other shard went from 14 GiB to 37.5 GiB: 9.46 GiB free, 14.19 GiB wanted, and
    three and a half hours of queue position lost to a traceback.  Nothing was
    wrong with either job -- they simply wanted the card at the same moment.
    Waiting is the right behaviour on a shared box, and a crash is not.

    The threshold is an environment variable rather than a figure derived from the
    checkpoint, because deriving it means reading a config and guessing a parameter
    count, and guessing wrong fails in the direction this function exists to
    prevent.  16 GiB covers the 7B teacher in bfloat16; G4's 8B needs it raised.
    """
    if not (torch.cuda.is_available() and WAIT_FOR_VRAM):
        return
    need = MIN_FREE_GIB if min_free_gib is None else min_free_gib
    import time
    t0, last = time.time(), -1e9
    while True:
        avail = free_vram_gib()
        if avail >= need:
            if time.time() - t0 > 30:
                print(f"    [vram] room appeared after "
                      f"{(time.time() - t0) / 60:.1f} min: {avail:.1f} GiB free",
                      flush=True)
            return
        waited = (time.time() - t0) / 60
        if waited > WAIT_TIMEOUT_MIN:
            raise RuntimeError(
                f"waited {waited:.0f} min for {need:.1f} GiB of free VRAM"
                f"{' to load ' + tag if tag else ''}; only {avail:.1f} GiB free. "
                f"Another process is holding the card. Raise SCD_VRAM_WAIT_MIN, "
                f"lower SCD_MIN_FREE_GIB, or wait for the neighbour to finish.")
        if time.time() - last > 300:
            last = time.time()
            print(f"    [vram] waiting for {need:.1f} GiB free "
                  f"({avail:.1f} available){' to load ' + tag if tag else ''}; "
                  f"{waited:.0f} of {WAIT_TIMEOUT_MIN:.0f} min elapsed", flush=True)
        time.sleep(20)


def load_model(name: str, train: bool = False, dtype=None,
               full_finetune: bool = False):
    """Load a causal LM.

    ``full_finetune`` switches to float32, and it is not a preference.  AdamW on
    bfloat16 weights with no float32 master copy loses small updates to rounding;
    with adapters that does not matter because the adapter is tiny and its updates
    are large, but when every weight is trainable it does.  A 0.5B student in
    float32 is 2 GB of weights, 2 GB of gradients and 4 GB of optimiser state.
    """
    from transformers import AutoModelForCausalLM
    dev = device()
    wait_for_vram(tag=name.split("/")[-1])
    dtype = dtype or (torch.float32 if full_finetune else torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="sdpa",
        device_map={"": 0} if dev == "cuda" else None,
    )
    model.config.use_cache = not train
    if not train:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model if dev == "cuda" else model.to(dev)


def attach_lora(model, r: int = 32, alpha: int | None = None,
                dropout: float = 0.0):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    m = get_peft_model(model, cfg)
    m.print_trainable_parameters()
    return m


def stop_ids(tok) -> set:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    unk = getattr(tok, "unk_token_id", None)
    for sym in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        i = tok.convert_tokens_to_ids(sym)
        if isinstance(i, int) and i >= 0 and i != unk and \
                tok.convert_ids_to_tokens(i) == sym:
            ids.add(i)
    return ids


# ------------------------------------------------------------------- releasing


def free(*objs, quiet: bool = False):
    """Actually release a model's VRAM, and say how much came back.

    The first version of this function was ``for o in objs: del o``, which frees
    nothing: ``del`` unbinds the helper's own local name, never the caller's.  The
    caller still held ``teacher``, so a 14.2 GiB model stayed resident for the rest
    of the stage.  Two budget failures were that bug -- 23.9 GiB and 21.9 GiB peaks
    on a 0.5B student with adapters -- and the arithmetic matched once a breakdown
    was printed: 14.19 (teacher) plus 0.99 (a previous student) was exactly the
    15.18 GiB of live memory present before training allocated anything of its own.

    The same mistake is recorded as M1 in an earlier project's notes in this
    repository.  Writing it again is the argument for the report below: a leak that
    prints its own size cannot survive a second run.

    The module's storages are therefore emptied in place.  That works whether or not
    the caller drops its reference, which is the only version of this function safe
    to call from twenty-two places.
    """
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    for o in objs:
        if o is None:
            continue
        try:
            for prm in o.parameters(recurse=True):
                prm.grad = None
                prm.data = torch.empty(0, device="cpu")
            for buf in o.buffers(recurse=True):
                buf.data = torch.empty(0, device="cpu")
        except AttributeError:
            pass          # not a module: dropping the caller's name is their job
    _remember_peak()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        if not quiet and objs:
            g = 2 ** 30
            print(f"    [free] released {(before - after) / g:.2f} GiB, "
                  f"{after / g:.2f} GiB still live", flush=True)
        torch.cuda.reset_peak_memory_stats()


def model_gib(model) -> float:
    """Resident size of a module's parameters and buffers, in GiB."""
    try:
        n = sum(p.numel() * p.element_size() for p in model.parameters()) \
            + sum(b.numel() * b.element_size() for b in model.buffers())
    except AttributeError:
        return 0.0
    return n / 2 ** 30


def assert_sole_resident(model, tag: str = "", slack_gib: float = 4.0) -> None:
    """Fail now if an earlier model is still holding VRAM.

    This is a structural invariant of the program -- one large model resident at a
    time -- not a resource threshold, which is why it stays fatal while
    ``check_budget`` does not.  It exists because the same leak reached the server
    twice, and both times it surfaced twenty minutes into training as a budget
    failure rather than as the missing release it actually was.
    """
    if not torch.cuda.is_available():
        return
    live = torch.cuda.memory_allocated() / 2 ** 30
    own = model_gib(model)
    if live > own + slack_gib:
        raise RuntimeError(
            f"stale VRAM{' at ' + tag if tag else ''}: {live:.2f} GiB live but this "
            f"model is only {own:.2f} GiB. An earlier model was not released. Check "
            f"that every hf.load_model is matched by hf.free, and that the [free] "
            f"lines in the log show the memory coming back.")


# ------------------------------------------------------------------ accounting


def device_limit_gib() -> float:
    """What the device actually has, asked rather than assumed."""
    if not torch.cuda.is_available():
        return float("inf")
    return torch.cuda.get_device_properties(0).total_memory / 2 ** 30


def check_budget(tag: str = ""):
    """Record the high-water mark; warn once past the budget, do not stop.

    This used to raise, and that was the wrong call.  The evidence is a run it
    killed: peak allocated 21.9 GiB, peak reserved 22.0 GiB, against 24 GB granted
    by the scheduler.  Two gigabytes spare, nothing wrong with the hardware, and the
    job died because of a number set by hand in a shell script.

    Three separate mistakes were in that design.

    *The threshold was a guess.*  ``SCD_VRAM_GB`` was the request minus an arbitrary
    margin.  The device's real capacity can simply be asked for.

    *It watched a high-water mark, not pressure.*  ``max_memory_allocated`` is
    cumulative since the last reset, so one transient spike in the first micro-batch
    condemned a run whose memory would have been flat for three thousand steps.

    *CUDA already fails better than this.*  Running out raises
    ``torch.cuda.OutOfMemoryError``, in-process and catchable, reporting allocated,
    reserved, requested and fragmentation -- more than this function ever knew.  The
    scheduler SIGKILLs a process for exceeding *host* memory, which no GPU-side
    guard can see anyway.

    What the guard was good for was diagnosis, and that job has been taken over by
    things that do it properly: the ``[mem]`` breakdown in ``train``, the ``[free]``
    lines above, and ``assert_sole_resident``.

    Set ``SCD_VRAM_HARD=1`` to restore the old behaviour.
    """
    if not torch.cuda.is_available():
        return
    _remember_peak()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    if peak <= VRAM_BUDGET_GB:
        return
    reserved = torch.cuda.max_memory_reserved() / 2 ** 30
    msg = (f"peak allocated {peak:.1f} GiB over the stated budget "
           f"{VRAM_BUDGET_GB:.0f} GiB{' at ' + tag if tag else ''} "
           f"(reserved {reserved:.1f}, device has {device_limit_gib():.1f})")
    if HARD_LIMIT:
        raise RuntimeError(
            msg + ".\n  SCD_VRAM_HARD=1 is set, so this is fatal. Lower, in this "
            "order: GEN_BATCH, LOGIT_BATCH, HIDDEN_BATCH, GRAD_CHECKPOINTING=1, "
            "TRAIN_BATCH -- and if TRAIN_BATCH comes down, raise ACCUM by the same "
            "factor so the effective batch is unchanged.")
    if tag not in _warned:
        _warned.add(tag)
        print(f"    [vram] {msg}. Continuing: the device, not this number, is the "
              f"limit. Set SCD_VRAM_HARD=1 to make it fatal.", flush=True)


def peak_report() -> dict:
    """Memory high-water marks, for the resource accounting every stage reports."""
    if not torch.cuda.is_available():
        return {}
    _remember_peak()
    g = 2 ** 30
    return {"peak_allocated_gib": _peak_seen_bytes / g,
            "peak_allocated_since_last_free_gib":
                torch.cuda.max_memory_allocated() / g,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / g,
            "device_total_gib": device_limit_gib(),
            "stated_budget_gib": VRAM_BUDGET_GB}


def mem() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    a = torch.cuda.memory_allocated() / 2 ** 30
    p = torch.cuda.max_memory_allocated() / 2 ** 30
    return f"alloc {a:.2f} / peak {p:.2f} GiB (budget {VRAM_BUDGET_GB:.0f})"
