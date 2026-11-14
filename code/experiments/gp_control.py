r"""GP -- positive control.  Does the pipeline transfer capability at all?

This runs before everything, including the CPU gates, because it validates the
*dependent variable*.  Two rounds of earlier work were invalidated by skipping it:
one reported a rank correlation of +0.60 between its diagnostic and a capability
gain whose four values were all negative, and another ran eighty training arms in a
regime where every arm scored below the untrained student.

The configuration is chosen to be maximally favourable, not interesting: same
family, large capability gap, and the supervision recipe the literature reports
recovers 96% of private capability -- supervised fine-tuning on answer-verified
teacher traces.  If capability does not move here, nothing downstream can be
measured, and the fix is the pipeline rather than the theory.
"""

from __future__ import annotations

import argparse

import numpy as np

from scd import corpus, data, hf, runio, stats, train
from experiments import _common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--base-shots", type=int, default=4,
                    help="few-shot reference for the base model, so a "
                         "gain cannot be manufactured out of format learning")
    ap.add_argument("--prompts", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--lrs", default="1e-5 2e-5 5e-5 1e-4")
    ap.add_argument("--measure-teacher", type=int, default=1,
                    help="evaluate the teacher on the suite so Gamma and "
                         "private-value recovery are measured, not assumed")
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--suite-limit", type=int, default=0)
    ap.add_argument("--tag", default="gp_control")
    # Present so that a dry run can exercise this stage's command path untouched
    # by the GPU.
    ap.add_argument("--list-arms", action="store_true",
                    help="print the stage's arms and exit, touching no GPU")
    a = ap.parse_args()
    if a.list_arms:
        print("[gp] learning-rate scan: %s" % a.lrs)
        print("[gp] seeds at the chosen rate: %s" % a.seeds)
        print("[gp] %d training runs in total"
              % (len(a.lrs.split()) + len(a.seeds.split())))
        return

    rows = C.build_traces(a.teacher, a.prompts, a.samples, gen_batch=a.gen_batch)
    train_rows, probe_rows, fit_rows = C.probe_and_fit_rows(rows)
    problems, census = C.suite(a.suite_limit)
    cov = C.coverage_of(train_rows, rows)
    print(f"[gp] C2 of the identity measure = {cov['c2']:.3f} "
          f"(must be ~1: mu = d by construction)")

    # The base reference is the base model at its BEST available prompting.
    #
    # This matters because the student here is not instruction-tuned.  Scored
    # zero-shot it may fail to produce the answer cue at all, and a gain measured
    # against that would partly be the student learning a format -- the very
    # confound both surveys flag, where imitation transfers style far more readily
    # than substance.  Taking the maximum of the zero-shot and few-shot scores
    # charges the study for whatever the base model already knows, however it has
    # to be asked.
    base_model, tok = C.student(a.student, a.lora_r)
    refs = {"zero_shot_chat": C.score(base_model, tok, problems, a.eval_batch,
                                      shots=0)}
    if a.base_shots:
        refs["few_shot_chat"] = C.score(base_model, tok, problems, a.eval_batch,
                                        shots=a.base_shots)
        refs["few_shot_plain"] = C.score(base_model, tok, problems, a.eval_batch,
                                         shots=a.base_shots, plain=True)
    hf.free(base_model)
    best = max(refs, key=lambda k: refs[k]["accuracy"])
    base = refs[best]
    for k, v in refs.items():
        print(f"[gp] base {k:16s} {100*v['accuracy']:6.2f}%"
              + ("   <- reference" if k == best else ""))

    # Gamma, measured rather than assumed.  The first version of this stage did
    # not evaluate the teacher, which left its own pass criterion ill-posed: an
    # absolute threshold in accuracy points on a quantity whose ceiling is set by
    # the configuration says nothing until the ceiling is known.  One extra
    # evaluation pass on the teacher fixes that, and it also yields the metric
    # the distillation survey recommends -- the fraction of the teacher-minus-base
    # gap that the student recovers.
    teacher_acc = float("nan")
    if a.measure_teacher:
        tt = hf.load_tokenizer(a.teacher)
        tm = hf.load_model(a.teacher)
        ts = C.score(tm, tt, problems, a.eval_batch, shots=0)
        hf.free(tm)
        teacher_acc = ts["accuracy"]
        print(f"[gp] teacher accuracy {100*teacher_acc:.2f}%  ->  "
              f"Gamma = {100*(teacher_acc - base['accuracy']):.2f} points")

    # Checkpoint and resume.
    #
    # This stage trains seven times on the identity measure (190,654 supervised
    # steps, 5.1 times the factorial), which takes about thirty-one hours, longer
    # than most GPU queues allow in one job.  Writing the blob only at the end
    # meant a job killed on walltime lost everything it had computed; it now
    # writes after each step and reads back what is finished, so resubmitting
    # continues instead of starting over.
    prev = runio.load(a.tag) or {}
    runs = list(prev.get("runs") or [])
    lr_scan = dict(prev.get("lr_scan") or {})
    if runs or lr_scan:
        print(f"[gp] resuming: {len(lr_scan)} rates and {len(runs)} seeds done")

    def ckpt(partial: bool = True) -> None:
        runio.save(a.tag, {"args": vars(a), "census": census,
                           "suite_n": base["n"], "partial": partial,
                           "base_accuracy": base["accuracy"],
                           "base_per_item": base["per_item"],
                           "base_by_source": base["by_source"],
                           "base_shots_used": base["shots"],
                           "base_reference_prompting": best,
                           "teacher_accuracy": teacher_acc,
                           "lr_scan": lr_scan, "runs": runs})

    ckpt()      # metadata is durable before the first training run
    for lr in [float(x) for x in a.lrs.split()]:
        if str(lr) in lr_scan:
            print(f"[gp] skipping lr={lr:g} (already done)")
            continue
        # Seed the adapter too, not only the data order; see C.student.
        model, tok = C.student(a.student, a.lora_r, seed=0)
        cfg = train.TrainConfig(objective="ce", lr=lr, epochs=a.epochs,
                                batch=a.batch, accum=a.accum, lora_r=a.lora_r, seed=0)
        info = train.train(model, tok, train_rows, cfg, probe_rows=probe_rows,
                           fit_rows=fit_rows)
        sc = C.score(model, tok, problems, a.eval_batch)
        lr_scan[str(lr)] = {"accuracy": sc["accuracy"], **{k: info[k] for k in
                            ("displacement", "own_supervision_loss_drop")}}
        print(f"[gp] lr={lr:g}: {sc['accuracy']*100:.2f}%  "
              f"displacement={info['displacement']:.3e}")
        hf.free(model)
        ckpt()

    best_lr = max(lr_scan, key=lambda k: lr_scan[k]["accuracy"])
    print(f"[gp] chosen lr = {best_lr}")
    oom = []
    done_seeds = {int(r["seed"]) for r in runs}
    for seed in [int(s) for s in a.seeds.split()]:
        if seed in done_seeds:
            print(f"[gp] skipping seed {seed} (already done)")
            continue
        def one(seed=seed):
            # Seed the adapter too, not only the data order; see C.student.
            model, tok = C.student(a.student, a.lora_r, seed=seed)
            cfg = train.TrainConfig(objective="ce", lr=float(best_lr),
                                    epochs=a.epochs, batch=a.batch, accum=a.accum,
                                    lora_r=a.lora_r, seed=seed)
            info = train.train(model, tok, train_rows, cfg, probe_rows=probe_rows,
                               fit_rows=fit_rows)
            sc = C.score(model, tok, problems, a.eval_batch)
            hf.free(model)
            return info, sc
        got, err = C.oom_safe(f"gp seed {seed}", one)
        if err:
            oom.append({"seed": seed, "error": err})
            continue
        info, sc = got
        paired = stats.paired_bootstrap(sc["per_item"], base["per_item"])
        runs.append({"seed": seed, "accuracy": sc["accuracy"],
                     "by_source": sc["by_source"], "per_item": sc["per_item"],
                     "paired_vs_base": paired,
                     **{k: info[k] for k in
                        ("displacement", "probe_drift", "train_seconds",
                         "own_supervision_loss_before", "own_supervision_loss_after",
                         "own_supervision_loss_drop", "steps",
                         "n_supervised_positions")}})
        C.append_run(a.tag, runs[-1])
        ckpt()          # each seed costs about four hours; do not risk losing one
        print(f"[gp] seed {seed}: {sc['accuracy']*100:.2f}%  "
              f"gain {100*(sc['accuracy']-base['accuracy']):+.2f} pts")
    if not runs:
        raise RuntimeError("every seed failed; see the [oom] lines above")

    acc = np.array([r["accuracy"] for r in runs])
    gain = 100.0 * (acc.mean() - base["accuracy"])
    gamma = 100.0 * (teacher_acc - base["accuracy"])
    recovery = gain / gamma if np.isfinite(gamma) and gamma > 0 else float("nan")
    sigma_w = float(acc.std(ddof=1)) if len(acc) > 1 else float("nan")
    suite_mde = data.mde(base["n"])
    runio.save(a.tag, {
        # partial=False matters: the submission script reads this key to decide
        # whether the stage finished, and a blob without it is treated as partial,
        # so the job would start again from the beginning.
        "partial": False,
        "args": vars(a), "census": census, "suite_n": base["n"],
        "suite_mde_points": suite_mde,
        "base_accuracy": base["accuracy"], "base_by_source": base["by_source"],
        "base_per_item": base["per_item"], "base_shots_used": base["shots"],
        "base_reference_prompting": best,
        "base_references": {k: {"accuracy": v["accuracy"], "shots": v["shots"],
                                "plain": v["plain"], "by_source": v["by_source"]}
                            for k, v in refs.items()},
        "format_learning_headroom_points":
            100 * (base["accuracy"] - refs["zero_shot_chat"]["accuracy"]),
        "chosen_lr": float(best_lr), "lr_scan": lr_scan,
        "runs": [{k: v for k, v in r.items() if k != "per_item"} for r in runs],
        "per_item": {str(r["seed"]): r["per_item"] for r in runs},
        "gain_points": gain, "all_seeds_positive": bool((acc > base["accuracy"]).all()),
        "sigma_w": sigma_w,
        "seed_mde_points": 100 * 2.8 * sigma_w / max(len(acc), 1) ** 0.5,
        "headroom_ratio": gain / suite_mde if suite_mde > 0 else float("nan"),
        "identity_c2": cov, "oom_arms": oom,
        "teacher_accuracy": teacher_acc,
        "teacher_by_source": ts["by_source"] if a.measure_teacher else {},
        "gamma_points": gamma,
        "private_value_recovery": recovery,
    })
    print(f"\n[gp] gain = {gain:+.2f} points, sigma_w = {sigma_w:.4f}, "
          f"suite MDE = {suite_mde:.2f}, headroom = {gain/suite_mde:.1f}x")
    if np.isfinite(recovery):
        print(f"[gp] Gamma = {gamma:.2f} points, private value recovery = "
              f"{100*recovery:.1f}%")


if __name__ == "__main__":
    main()
