r"""G3 -- is it the objective or the measure?  The decisive intervention.

Held fixed: the text, the teacher, the teacher's cached logits, the number of
supervised positions, the number of optimiser steps.  Varied: the objective
(cross-entropy or forward KL) and the *distribution of supervised positions* over
the trace.  If coverage is the active variable, this alone produces a difference.

This replaces the argument from the field's 99%/56%/12%/96% figures.  Those cannot
be used as evidence here, because the supervision measure of the full-logit regime
in that study is not reported -- and if its corpus was teacher-sampled then its
measure equals the teacher's occupancy, the coefficient is 1, and the coverage
account explains nothing about it.  Reconstructing the regimes in-house with the
measure under explicit control is the only defensible version.

**The student sweep is not optional.**  The synthetic gate showed the theory
predicts a regime, not a main effect: where capacity binds, fidelity on the corpus
already predicts capability and coverage adds nothing; where capacity is ample,
fidelity goes blind and coverage carries the signal.  A single 0.5B student sits in
the capacity-limited regime, so the first version of this experiment -- one small
student -- was designed where the predicted effect is smallest.  Two student sizes
are run and the *interaction* is the pre-registered test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib

import numpy as np

from scd import corpus, data, hf, runio, stats, train
from scd import teacherlogits
from experiments import _common as C

MODES = ("uniform", "early", "late", "cluster_drop")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--students", default="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompts", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--modes", default=" ".join(MODES))
    ap.add_argument("--objectives", default="ce fkl")
    ap.add_argument("--seeds", default="0 1 2 3 4")
    ap.add_argument("--lr", type=float, default=0.0)      # 0 = read gp_control
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--logit-batch", type=int, default=4)
    ap.add_argument("--suite-limit", type=int, default=0)
    ap.add_argument("--tag", default="g3_measure_vs_loss")
    ap.add_argument("--min-recovery-spread", type=float, default=15.0,
                    help="points of Gamma-recovery the student sweep must span "
                         "before the main comparison is worth running")
    ap.add_argument("--sweep-students", type=int, default=2,
                    help="how many students carry the main comparison. The probe "
                         "runs on every student in --students; the sweep keeps the "
                         "ones that span the capacity axis most widely, so adding "
                         "a candidate costs one probe arm, not a whole sweep")
    ap.add_argument("--reverse", type=int, default=0,
                    help="walk the arm list from the end. A second process with "
                         "--reverse 1 on another GPU converges on the first from "
                         "the other side: no coordination, no stopping, and the "
                         "split lands in proportion to each GPU's actual speed")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    # The untrained score costs ~25 minutes of generation per student and feeds only
    # base_accuracy and gain_points.  It is deterministic -- greedy decoding, and
    # PEFT starts lora_B at zero so the adapter is a no-op before training -- so a
    # worker can be handed the number instead of re-deriving it.  Twenty shards
    # would otherwise spend sixteen hours measuring the same two constants.
    ap.add_argument("--base-accuracies", default="",
                    help="space-separated NAME=accuracy pairs, e.g. "
                         "'Qwen/Qwen2.5-0.5B=0.0665514 Qwen/Qwen2.5-1.5B=0.0322674'")
    ap.add_argument("--list-arms", action="store_true",
                    help="print this shard's arms and exit, touching no GPU")
    # Open the logit cache that ships with the package instead of building one.
    # Building needs the 15 GB teacher, and with twenty workers each would load it
    # again; on a compute node running offline, without the teacher weights, that
    # load does not merely take time, it fails. If the cache is genuinely missing,
    # train.train reports it at the arm that needs it.
    ap.add_argument("--cached-logits", type=int, default=0,
                    help="1 = open the existing logit cache instead of building it")
    ap.add_argument("--runs-tag", default="",
                    help="whose <tag>_runs.jsonl to append to and re-read. Point "
                         "every worker at ONE file and they deduplicate against "
                         "each other with no coordination protocol at all")
    ap.add_argument("--skip-done", default="",
                    help="a <tag>_runs.jsonl whose (student, objective, mode, "
                         "seed) arms are already finished and must not be redone")
    ap.add_argument("--skip-probe", type=int, default=0,
                    help="trust an earlier run's capacity probe instead of "
                         "spending ~6 GPU-h repeating it")
    # The capacity probe is a registered gate, not an optional extra: if the
    # recovery spread between students is too narrow the main comparison cannot be
    # read and the stage stops. With one job per shard it has to run exactly once
    # and be reused, rather than costing every worker about two and a half GPU
    # hours.
    #
    # The existing --skip-probe inherits it from disk, which works only when all
    # workers share one results directory. Here each job has its own, so passing
    # the numbers in is both simpler and visible in the log. The probe computes
    # recovery = (floor - base) / (teacher_acc - base), and reads teacher_acc from
    # the gp_control blob at a fixed path. On a cluster that blob may not be there
    # yet, in which case recovery is nan, the gate does not open and every shard
    # stops. Passing the values in removes both the dependency and the ordering.
    ap.add_argument("--teacher-accuracy", type=float, default=None,
                    help="teacher accuracy, in place of reading the gp_control blob")
    ap.add_argument("--probe-only", type=int, default=0,
                    help="stop after the capacity probe, before the main comparison")
    ap.add_argument("--probe-recovery", default="",
                    help="NAME=recovery pairs from a probe that already ran, used "
                         "in place of measuring it again")
    ap.add_argument("--force-full", type=int, default=0,
                    help="run the main comparison even if the capacity probe says "
                         "the student contrast is too narrow to interpret")
    a = ap.parse_args()

    modes = a.modes.split()
    objs = a.objectives.split()
    seeds = [int(s) for s in a.seeds.split()]
    students = a.students.split()

    if a.list_arms:
        for st in students:
            arms = list(itertools.product(objs, modes, seeds))
            if a.reverse:
                arms.reverse()
            if a.nshards > 1:
                arms = [x for i, x in enumerate(arms)
                        if i % a.nshards == a.shard]
            print("[g3] %s shard %d/%d: %d arm(s)"
                  % (st, a.shard, a.nshards, len(arms)))
            for i, (obj, mode, seed) in enumerate(arms):
                print("  %2d  objective=%-4s mode=%-13s seed=%d"
                      % (i, obj, mode, seed))
        return

    base_rows = C.build_traces(a.teacher, a.prompts, a.samples, gen_batch=a.gen_batch)
    problems, census = C.suite(a.suite_limit)
    cache = None
    if any(o != "ce" for o in objs):
        if a.cached_logits:
            d = C.logits_dir(a.teacher)
            # The zip bundle first, the directory second: the 15,994 .npz files
            # are collapsed into one because the cluster quota counts files.
            z = d.with_suffix(".zip")
            if z.is_file():
                import zipfile
                with zipfile.ZipFile(z) as zf:
                    n = len(zf.namelist())
                print(f"[g3] using the logit-cache bundle: {n} members in {z.name}")
                cache = teacherlogits.Cache(z)
            else:
                n = sum(1 for _ in d.glob("*.npz")) if d.is_dir() else 0
                assert n, f"--cached-logits but neither {z} nor {d} is present"
                print(f"[g3] using the existing logit cache: {n} files in {d}")
                cache = teacherlogits.Cache(d)
        else:
            cache, _ = C.build_logit_cache(a.teacher, base_rows,
                                           micro_batch=a.logit_batch)

    # ---- V4.6: confirm the intervention actually moves C2, before any training
    full = C.coverage_of(base_rows, base_rows)
    prepared, cov = {}, {"full": full}

    # Two passes, because the count has to be matched DOWN to what every mode can
    # actually reach rather than up to a target one of them cannot.
    #
    # The first run asserted its way out here, correctly: three modes hit the
    # 47,664-position target and cluster_drop reached 37,509.  That is not a bug in
    # the check -- V4.1 exists precisely so that a data-quantity difference cannot
    # masquerade as a coverage difference -- it is a bug in the target.
    # ``cluster_drop`` keeps only the cells that survive banning rarest-first by
    # mass, and the surviving cells simply do not contain 47,664 supervised steps;
    # ``match_supervised_count`` can turn positions off but never on, so no amount
    # of matching reaches a target above a mode's ceiling.
    #
    # Taking the floor instead costs every arm 21% of its supervised positions and
    # biases nothing: the comparison is across modes at one matched count, and
    # which count that is only moves power, not the contrast.
    raw = {}
    for mode in modes:
        rows = corpus.load(C.traces_path(a.teacher, a.prompts, a.samples))
        corpus.mask_fraction(rows, a.frac, mode, seed=0)
        raw[mode] = rows
    ceilings = {m: sum(r.n_supervised_steps() for r in raw[m]) for m in modes}
    target = min(min(ceilings.values()),
                 int(round(a.frac * full["n_supervised_steps"])))
    print(f"[g3] per-mode ceilings {ceilings}; matching all modes down to {target}")
    for mode in modes:
        got = corpus.match_supervised_count(raw[mode], target, seed=0)
        prepared[mode] = raw[mode]
        cov[mode] = C.coverage_of(raw[mode], base_rows)
        print(f"[g3] mode={mode:12s} C2={cov[mode]['c2']:8.3f} "
              f"supervised={got} (target {target})")
    c2s = [cov[m]["c2"] for m in modes]
    spread = max(c2s) / max(min(c2s), 1e-9)
    print(f"[g3] C2 spread across modes = {spread:.2f}x "
          f"(validity condition V4.6 needs >= 1.5x)")
    counts = [cov[m]["n_supervised_steps"] for m in modes]
    assert max(counts) - min(counts) <= 0.01 * max(counts) + 5, \
        f"supervised counts not matched across modes: {counts} (V4.1)"

    # Fail fast on what the stages below assume, before any GPU work.
    assert all(m in prepared for m in modes), \
        f"prepared corpora missing: {[m for m in modes if m not in prepared]}"
    assert base_rows and all(r.n_supervised_steps() > 0 for r in base_rows[:50]), \
        "base_rows is not the identity measure; the capacity probe needs mu = d"

    lr_default = a.lr
    if lr_default == 0.0:
        gp = runio.load("gp_control", where="results")
        lr_default = float(gp["chosen_lr"]) if gp else 1e-4
        print(f"[g3] learning rate from gp_control: {lr_default:g}")

    runs_tag = a.runs_tag or a.tag
    ledger = runio.results_dir() / f"{runs_tag}_runs.jsonl"

    want = os.environ.get("SCD_REQUIRE_CODE_SHA") == "1"

    def done_arms() -> set:
        """Arms already finished, re-read live so workers see each other.

        The file is append-only and every record is one short line, so a reader
        can race a writer only on the final line; a torn last line is skipped
        rather than crashing the worker that found it.
        """
        out = set()
        if not ledger.exists():
            return out
        # SCD_REQUIRE_CODE_SHA=1, set for a training job and not for a merge:
        # inherit only the arms this revision wrote. A record with no code_sha, or
        # with a different one, belongs to an earlier run; skipping it means it is
        # trained again instead of being mixed silently into the results. A merge
        # must accept every record, so it does not set the variable.
        mine = runio.code_sha() if want else None
        foreign = 0
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a writer was mid-append; the arm will reappear
            if want and r.get("code_sha") != mine:
                foreign += 1
                continue
            out.add((r["student"], r["objective"], r["mode"], r["seed"]))
        if foreign:
            print("[g3] %d arms in %s were written by a different revision "
                  "(code_sha != %s); they will be retrained rather than inherited"
                  % (foreign, ledger.name, mine))
        return out

    done = done_arms()
    if done:
        print(f"[g3] {len(done)} arms already in {ledger.name}; they will be skipped")

    # ------------------------------------------------- capacity probe (G3.0)
    #
    # The synthetic stage found that coverage predicts the capability gap only
    # where capacity is ample: at the tightest capacity the product carried
    # rho = +0.01 against fidelity's +0.26, and only from the second-largest
    # capacity upward did it reverse to +0.65 against -0.00.  So this comparison
    # is interpretable only if the student sweep actually spans that axis.
    #
    # It may not.  Stage GP measured the 0.5B student recovering 47.5% of a
    # 65.6-point gap, which is the capacity-limited end -- the regime where the
    # theory predicts no coverage effect.  Running the full comparison there and
    # finding nothing would not distinguish "the theory is wrong" from "it was
    # tested in the wrong regime", which is the one ambiguity this whole ladder
    # exists to prevent.
    #
    # So: one seed of the identity measure per student first, recovery computed
    # against the teacher, and the stage stops if the contrast is too narrow.
    # Two hours spent to decide whether to spend twenty-five.
    gp = runio.load("gp_control", where="results") or {}
    teacher_acc = (a.teacher_accuracy if a.teacher_accuracy is not None
                   else gp.get("teacher_accuracy"))
    if a.teacher_accuracy is not None:
        print(f"[g3] teacher accuracy {teacher_acc:.4f} (passed in, no blob read)")
    runs, bases, oom = [], {}, []

    # Load the finished arms INTO `runs`, not merely into the set used to skip
    # them. The other stages do this; this one did not, and the consequence
    # surfaced only at the merge: with every arm already in the ledger the loop
    # skipped all of them, `runs` stayed empty, and the merged blob came out with
    # an empty anova, an empty contrast and nan everywhere, while all eighty arms
    # sat intact in the ledger.
    for line in (ledger.read_text().splitlines() if ledger.exists() else []):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if want and r.get("code_sha") != runio.code_sha():
            continue          # same rule as done_arms: do not mix revisions
        runs.append(r)
    if runs:
        print(f"[g3] resumed {len(runs)} arms from {ledger.name}")

    probe_rec = {}
    given = {}
    for pair in a.base_accuracies.split():
        name, _, val = pair.partition("=")
        given[name] = float(val)

    for st in students:
        if st in given:
            # The same shape C.score returns, even where there is no value for a
            # key. A missing `by_source` made the checkpoint fail with a KeyError
            # after the gate had opened, which is after the job had done all of its
            # CPU work. Write None explicitly, with a `given` flag so the blob
            # records that the number was passed in and not measured here.
            b = {"accuracy": given[st], "by_source": None, "per_item": None,
                 "given": True}
            bases[st] = b
            print(f"[g3] {st}: base {b['accuracy']*100:.2f}% (given, not measured)")
        else:
            m, tok = C.student(st, a.lora_r)
            b = C.score(m, tok, problems, a.eval_batch)
            bases[st] = b
            print(f"[g3] {st}: base {b['accuracy']*100:.2f}%")
            hf.free(m)
        # --probe-recovery means the recovery was measured by a separate probe
        # job and passed in; measuring it again here pays for it twice. Without
        # this line every shard trained the floor arm for both students and only
        # then overwrote the result with the value passed in: about two and a half
        # GPU hours discarded per shard. The gate is still scored, on the probe's
        # measurement.
        if a.skip_probe or a.probe_recovery:
            continue

        def floor_arm(st=st):
            # base_rows, not prepared["full"]: the probe has to train on the
            # IDENTITY measure (every step supervised, mu = d), and "full" exists
            # only as a coverage reading in `cov`, never as a prepared corpus.
            # The first run crashed here with KeyError after ten minutes of base
            # evaluation -- a missing key that a one-line assert at startup would
            # have caught before the GPU was touched, which is why there is one now.
            tr, pr, fi = C.probe_and_fit_rows(base_rows)
            # Seed the adapter too, not only the data order; see C.student.
            model, tk = C.student(st, a.lora_r, seed=int(seeds[0]))
            cfg = train.TrainConfig(objective="ce", lr=lr_default, epochs=a.epochs,
                                    batch=a.batch, accum=a.accum,
                                    lora_r=a.lora_r, seed=int(seeds[0]))
            train.train(model, tk, tr, cfg, cache=cache, probe_rows=pr, fit_rows=fi)
            sc = C.score(model, tk, problems, a.eval_batch)
            hf.free(model)
            return sc
        got, err = C.oom_safe(f"g3 capacity probe {C.slug(st)}", floor_arm)
        if err:
            oom.append({"student": st, "stage": "capacity_probe", "error": err})
            continue
        gam = (teacher_acc - b["accuracy"]) if teacher_acc else float("nan")
        rec = (got["accuracy"] - b["accuracy"]) / gam if gam and gam > 0 else float("nan")
        probe_rec[st] = {"base": b["accuracy"], "floor": got["accuracy"],
                         "gamma_points": 100 * gam if gam == gam else None,
                         "recovery": rec}
        print(f"[g3] {C.slug(st)}: identity-measure floor "
              f"{100*got['accuracy']:.2f}%, recovery of Gamma "
              f"{100*rec:.1f}%" if rec == rec else f"[g3] {C.slug(st)}: floor "
              f"{100*got['accuracy']:.2f}% (no teacher accuracy on record)")

    # Keep the students that span the axis most widely, and sweep those.  This is
    # selection on a quantity measured BEFORE the comparison and belonging to the
    # (teacher, student) pair -- the same logic as G4's Gamma filter -- not
    # selection on the contrast under test.  Registered here before G3 runs.
    ranked = sorted((v["recovery"], k) for k, v in probe_rec.items()
                    if v["recovery"] == v["recovery"])
    if len(ranked) > a.sweep_students >= 2:
        keep = {ranked[0][1], ranked[-1][1]}
        for _, k in reversed(ranked[1:-1]):
            if len(keep) >= a.sweep_students:
                break
            keep.add(k)
        dropped = [k for k in students if k not in keep]
        students = [k for k in students if k in keep]
        print(f"[g3] sweep keeps {[C.slug(s) for s in students]}; probed but not "
              f"swept: {[C.slug(s) for s in dropped]}")

    recs = [v["recovery"] for v in probe_rec.values() if v["recovery"] == v["recovery"]]
    # probe_spread, not spread: `spread` already holds the C2 spread ratio measured
    # at line 136 and the final save reports it as c2_spread_ratio.
    probe_spread = 100 * (max(recs) - min(recs)) if len(recs) >= 2 else float("nan")
    if a.probe_recovery:
        probe_rec = {}
        for pair in a.probe_recovery.split():
            name, _, val = pair.partition("=")
            probe_rec[name] = {"recovery": float(val), "given": True}
        recs = [v["recovery"] for v in probe_rec.values()]
        probe_spread = (100 * (max(recs) - min(recs)) if len(recs) >= 2
                        else float("nan"))
        print(f"[g3] capacity probe given, not measured: spread "
              f"{probe_spread:.1f} pts over {len(recs)} students")
    elif a.skip_probe:
        prev = runio.load(a.tag.rsplit("_shard", 1)[0], where="results") or {}
        probe_rec = (prev.get("capacity_probe") or {}).get("per_student", {}) or {
            st: {"recovery": float("nan")} for st in students}
        recs = [v["recovery"] for v in probe_rec.values()
                if v.get("recovery") == v.get("recovery")]
        probe_spread = (100 * (max(recs) - min(recs)) if len(recs) >= 2
                        else float("nan"))
        print(f"[g3] capacity probe skipped; inherited spread {probe_spread:.1f} pts")
    probe = {"per_student": probe_rec, "swept_students": list(students),
             # "not measured in this invocation", true of both --skip-probe,
             # which inherits from disk, and --probe-recovery, which is passed in.
             # A merge prefers a blob that measured the probe itself, so this flag
             # has to say which one did.
             "inherited": bool(a.skip_probe or a.probe_recovery),
             "given": bool(a.probe_recovery),
             "recovery_spread_points": probe_spread,
             "required_spread_points": a.min_recovery_spread,
             "wide_enough": bool(probe_spread == probe_spread
                                 and probe_spread >= a.min_recovery_spread)}
    print(f"\n[g3] capacity probe: recovery spans {probe_spread:.1f} points across "
          f"{len(recs)} students (need >= {a.min_recovery_spread})")
    if a.probe_only:
        runio.save(a.tag, {"args": vars(a), "capacity_probe": probe,
                           "coverage": cov, "bases":
                           {k: v["accuracy"] for k, v in bases.items()},
                           "oom_arms": oom, "runs": [],
                           "probe_only": True})
        print("[g3] --probe-only: the probe is finished and written; stopping.")
        print("[g3] pass it to the workers with --probe-recovery:")
        print("     " + " ".join("%s=%.6f" % (k, v["recovery"])
                                 for k, v in probe_rec.items()
                                 if v.get("recovery") == v.get("recovery")))
        return

    if not probe["wide_enough"] and not a.force_full:
        runio.save(a.tag, {"args": vars(a), "capacity_probe": probe,
                           "coverage": cov, "bases":
                           {k: v["accuracy"] for k, v in bases.items()},
                           "oom_arms": oom, "runs": [],
                           "aborted": "capacity probe too narrow"})
        print("[g3] STOPPING before the main comparison: the student sweep does not "
              "span the capacity axis the effect is predicted to live on.\n"
              "     Add a larger student (Qwen2.5-3B) or a stronger teacher, then "
              "rerun. --force-full 1 overrides.")
        return

    # Save the blob after every arm, not only when the sweep ends.  An arm costs
    # about an hour, and a worker killed by an OOM, a node reboot or a relaunch used
    # to leave nothing usable behind: its per-arm records survived in the shared
    # ledger, but the startup metadata the gate reads (the MDE, the C2 table, the
    # bases, the capacity probe) and the per-item outcome vectors lived only in
    # memory until the final save.  runio.save writes a temp file and renames it
    # into place, so a kill during the write cannot leave a truncated blob.
    #
    # The checkpoint is marked partial: it is insurance and merge_g3.py input, never
    # a gate input, because a single worker only ever holds a fraction of the design
    # and its inline analysis would be computed on that fraction.
    def checkpoint(partial: bool = True) -> None:
        runio.save(a.tag, {
            "args": vars(a), "census": census, "suite_n": len(problems),
            "suite_mde_points": data.mde(len(problems)),
            "coverage": cov, "c2_spread_ratio": spread,
            "base": {k: {"accuracy": v["accuracy"],
                         "by_source": v.get("by_source"),
                         "given": bool(v.get("given"))}
                     for k, v in bases.items()},
            "base_per_item": {k: v.get("per_item") for k, v in bases.items()},
            "runs": [{k: v for k, v in r.items() if k != "per_item"} for r in runs],
            # Only arms that still hold per_item in memory. An arm reloaded from
            # the ledger does not: append_run drops the key so the ledger stays
            # small. Indexing r["per_item"] directly made the merge fail with a
            # KeyError immediately after it had reloaded all eighty arms.
            "per_item": {f"{C.slug(r['student'])}|{r['objective']}|{r['mode']}|"
                         f"{r['seed']}": r["per_item"]
                         for r in runs if r.get("per_item") is not None},
            "oom_arms": oom, "mode_ceilings": ceilings, "matched_count": target,
            "capacity_probe": probe, "partial": partial,
            "arms_this_worker": len(runs)})

    checkpoint()      # metadata is durable before the first arm is trained
    for st in students:
        b = bases[st]
        arms = list(itertools.product(objs, modes, seeds))
        if a.reverse:
            arms.reverse()
        if a.nshards > 1:
            arms = [x for i, x in enumerate(arms) if i % a.nshards == a.shard]
        for obj, mode, seed in arms:
            # Live check, not a startup snapshot: three workers walking the same
            # list from both ends need to see each other's finished arms, and the
            # shared ledger is the only state they share.
            if (st, obj, mode, int(seed)) in done_arms():
                print(f"[g3] skip {C.slug(st)} {obj} {mode} s{seed} (done elsewhere)",
                      flush=True)
                continue
            def one(obj=obj, mode=mode, seed=seed, st=st):
                tr, probe, fit = C.probe_and_fit_rows(prepared[mode])
                # Seed the adapter too, not only the data order; see C.student.
                model, tok = C.student(st, a.lora_r, seed=seed)
                cfg = train.TrainConfig(objective=obj, lr=lr_default,
                                        epochs=a.epochs, batch=a.batch,
                                        accum=a.accum, lora_r=a.lora_r, seed=seed)
                info = train.train(model, tok, tr, cfg, cache=cache,
                                   probe_rows=probe, fit_rows=fit)
                sc = C.score(model, tok, problems, a.eval_batch)
                hf.free(model)
                return info, sc
            got, err = C.oom_safe(f"g3 {C.slug(st)} {obj} {mode} s{seed}", one)
            if err:
                oom.append({"student": st, "objective": obj, "mode": mode,
                            "seed": seed, "error": err})
                continue
            info, sc = got
            runs.append({"student": st, "objective": obj, "mode": mode, "seed": seed,
                         "accuracy": sc["accuracy"], "per_item": sc["per_item"],
                         "by_source": sc["by_source"],
                         "c2": cov[mode]["c2"],
                         **{k: info[k] for k in
                            ("displacement", "probe_drift", "steps",
                             "n_supervised_positions", "own_supervision_loss_drop",
                             "train_seconds")}})
            C.append_run(runs_tag, runs[-1])
            checkpoint()
            print(f"[g3] {C.slug(st)} {obj:4s} {mode:12s} s{seed}: "
                  f"{sc['accuracy']*100:.2f}%  (base {b['accuracy']*100:.2f}%)")

    # ---------------------------------------------------------------- analysis
    def cell(st, obj, mode):
        return [r["accuracy"] for r in runs
                if r["student"] == st and r["objective"] == obj and r["mode"] == mode]

    anova = {}
    for st in students:
        vals = {(o, m): cell(st, o, m) for o in objs for m in modes
                if cell(st, o, m)}
        anova[st] = stats.eta_squared(vals) if len(vals) >= 4 else {}

    contrast = {}
    for st in students:
        for obj in objs:
            u = cell(st, obj, "uniform")
            worst_mode = min([m for m in modes if m != "uniform"],
                             key=lambda m: np.mean(cell(st, obj, m)) if cell(st, obj, m) else 1)
            w = cell(st, obj, worst_mode)
            if u and w:
                contrast[f"{C.slug(st)}|{obj}"] = {
                    "worst_mode": worst_mode,
                    "uniform_minus_worst_points": 100 * (np.mean(u) - np.mean(w)),
                    "seed_test": stats.seed_bootstrap(u, w),
                    "item_test": _item_test(runs, st, obj, worst_mode)}

    # the pre-registered interaction: the coverage effect should be larger for the
    # larger (more capable) student
    inter = {}
    if len(students) >= 2:
        for obj in objs:
            eff = []
            for st in students:
                k = f"{C.slug(st)}|{obj}"
                eff.append(contrast[k]["uniform_minus_worst_points"] if k in contrast
                           else float("nan"))
            inter[obj] = {"students": students, "effect_points": eff,
                          "interaction_points": eff[-1] - eff[0]}

    rank = stats.spearman([cov[m]["c2"] for m in modes],
                          [np.mean([r["accuracy"] for r in runs if r["mode"] == m])
                           for m in modes])

    runio.save(a.tag, {
        "args": vars(a), "census": census, "suite_n": len(problems),
        "suite_mde_points": data.mde(len(problems)),
        "coverage": cov, "c2_spread_ratio": spread,
        "base": {k: {"accuracy": v["accuracy"],
                     "by_source": v.get("by_source"),
                     "given": bool(v.get("given"))}
                 for k, v in bases.items()},
        "base_per_item": {k: v.get("per_item") for k, v in bases.items()},
        "runs": [{k: v for k, v in r.items() if k != "per_item"} for r in runs],
        "per_item": {f"{C.slug(r['student'])}|{r['objective']}|{r['mode']}|{r['seed']}":
                     r["per_item"] for r in runs if r.get("per_item") is not None},
        "anova": anova, "contrast": contrast, "interaction": inter,
        "rank_c2_vs_accuracy": rank, "oom_arms": oom, "partial": False,
        "mode_ceilings": ceilings, "matched_count": target,
        "capacity_probe": probe,
        "reliability": stats.reliability(
            [cell(st, o, m) for st in students for o in objs for m in modes
             if len(cell(st, o, m)) >= 2]),
    })

    print("\n--- G3 summary -------------------------------------------------")
    for st in students:
        e = anova.get(st, {})
        if e:
            print(f"  {C.slug(st)}: eta2 loss={e['eta2_loss']:.3f} "
                  f"measure={e['eta2_measure']:.3f} "
                  f"interaction={e['eta2_interaction']:.3f}")
    for k, v in contrast.items():
        # item_test may not exist, since the ledger carries no per_item. This
        # line has to tolerate that: the blob is already saved by the time it
        # runs, so a KeyError in the summary turns a successful merge into a
        # reported failure and sends the reader looking for a fault that is not
        # there.
        it = v["item_test"]
        item = ("item p=%.3f" % it["p"]) if "p" in it else "item: not available"
        print(f"  {k}: uniform - {v['worst_mode']} = "
              f"{v['uniform_minus_worst_points']:+.2f} pts "
              f"(seed p={v['seed_test']['p']:.3f}, {item})")
    for obj, v in inter.items():
        print(f"  interaction[{obj}]: {v['effect_points']} -> "
              f"{v['interaction_points']:+.2f} pts larger for the bigger student")


def _item_test(runs, st, obj, worst_mode):
    """The paired item-level test, or the reason it cannot be run.

    It needs the per-item outcome vector of every arm. That vector lives in the
    shard blob and not in the ledger, so when a merge has only the ledger the test
    does not exist, and the blob has to say so instead of passing over it or
    producing an empty number.
    """
    a = _mean_items(runs, st, obj, "uniform")
    b = _mean_items(runs, st, obj, worst_mode)
    if not a or not b:
        return {"available": False,
                "why": ("per_item is not in the ledger; the item-level test needs "
                        "the shard blobs. seed_test above is unaffected.")}
    out = dict(stats.paired_bootstrap(a, b))
    out["available"] = True
    return out


def _mean_items(runs, st, obj, mode):
    """Per-item outcome averaged over seeds, then thresholded for a paired test."""
    # An arm reloaded from the ledger carries no per_item, so the item-level test
    # is available only where the shard blobs still hold the vector.
    sel = [r["per_item"] for r in runs if r["student"] == st
           and r["objective"] == obj and r["mode"] == mode
           and r.get("per_item") is not None]
    return (np.mean(np.array(sel, float), 0) > 0.5).astype(int).tolist() if sel else []


if __name__ == "__main__":
    main()
