#!/usr/bin/env python
"""Combine G3's sharded workers into one analysed result.

G3 was split across three processes -- one walking the arm list forward on GPU 0
and two walking it backward on GPU 1 -- so the arms land in three result blobs and
one shared ledger.  This is the authoritative analysis: it unions the arms,
deduplicates them, and writes the canonical ``results/g3_measure_vs_loss.json``
that the gate and the tables read.

Why the workers do not analyse.  Each one sees only its own share, so each one's
inline ANOVA is computed on a fraction of the design; the gate must never read one
of those.  Running this script is the step that turns three partial sweeps into one
experiment, and it refuses to pretend otherwise when arms are missing.

Why the ledger is also read.  A worker writes its blob only when its sweep ends,
but it appends to the shared ledger after every arm.  So a worker that is still
running -- or one that is killed before it finishes -- contributes nothing through
its blob while its completed arms sit safely in the ledger.  Reading both means no
finished arm is ever lost, and the output records which arms arrived that way.
Ledger rows carry every field the analysis needs except the per-item outcome
vectors, so an arm known only from the ledger drops out of the item-level paired
test (the seed-level test is unaffected).

    python merge_g3.py                       # merge, analyse, write the canonical blob
    python merge_g3.py --dry-run             # report coverage of the design only
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from scd import data, runio, stats

ROOT = pathlib.Path(__file__).resolve().parent


def slug(name: str) -> str:
    return name.split("/")[-1].replace(".", "p")


def _c2_spread(cov, modes):
    """The C2 spread the intervention achieved, from the coverage table itself."""
    c2s = [cov[m]["c2"] for m in modes if m in cov and "c2" in cov[m]]
    if len(c2s) < 2:
        return float("nan")
    return max(c2s) / max(min(c2s), 1e-9)


def _dedup_oom(blobs, completed=frozenset()):
    """OOM arms that are still missing from the design.

    Two things have to be filtered out.  Two workers walking the list from both
    ends can both hit the same arm, and the same arm can appear in a worker blob
    and in an earlier merged blob, so the list is deduplicated by the arm it names.
    And an arm that one worker lost to an OOM but another later completed is not a
    lost arm -- it is in ``completed`` and belongs nowhere in this list.  Without
    that second filter the gate keeps reporting a hole that has since been filled.
    """
    marked, out = set(), []
    for tag, d in sorted(blobs.items()):
        for o in (d.get("oom_arms") or []):
            arm = (o.get("student"), o.get("objective"), o.get("mode"),
                   o.get("seed"))
            if arm in completed:
                continue
            key = arm + (o.get("stage"),)
            if key in marked:
                continue
            marked.add(key)
            out.append(dict(o, _shard=tag))
    return out


def _mean_items(per_item, runs, st, obj, mode):
    """Per-item outcome averaged over seeds, then thresholded for a paired test."""
    keys = [f"{slug(r['student'])}|{r['objective']}|{r['mode']}|{r['seed']}"
            for r in runs
            if r["student"] == st and r["objective"] == obj and r["mode"] == mode]
    sel = [per_item[k] for k in keys if k in per_item]
    if not sel:
        return []
    n = min(len(s) for s in sel)
    return (np.mean([s[:n] for s in sel], axis=0) >= 0.5).astype(int).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="g3_measure_vs_loss g3_measure_vs_loss_rev0 "
                                      "g3_measure_vs_loss_rev1 g3_measure_vs_loss_fwd "
                                      "g3_measure_vs_loss_fill")
    ap.add_argument("--out", default="g3_measure_vs_loss")
    ap.add_argument("--ledger", default="g3_measure_vs_loss_runs.jsonl",
                    help="the shared append-only ledger the workers write after "
                         "every arm; its rows fill in for any worker whose blob "
                         "does not exist yet.  Empty string disables it.")
    ap.add_argument("--dry-run", type=int, default=0)
    a = ap.parse_args()

    blobs = {}
    for tag in a.tags.split():
        if tag == a.out:
            # The canonical output shares a name with the forward worker's tag.
            # Reading it back would re-attribute every arm to this script and
            # double-count the per-blob fields, so skip it and take those arms
            # from the ledger instead.
            continue
        d = runio.load(tag, where="results")
        if d is not None:
            blobs[tag] = d

    runs, per_item, seen = [], {}, set()
    for tag, d in blobs.items():
        for r in d.get("runs", []):
            key = (r["student"], r["objective"], r["mode"], r["seed"])
            if key in seen:
                continue
            seen.add(key)
            r = dict(r); r["_shard"] = tag
            runs.append(r)
        per_item.update(d.get("per_item", {}))

    # A blob takes precedence because it carries the per-item vectors; the ledger
    # then supplies the arms no finished blob covers.  The last line can be torn if
    # a worker is appending as we read, so a row that will not parse is dropped.
    from_ledger = 0
    led = runio.results_dir() / a.ledger if a.ledger else None
    if led is not None and led.exists():
        for line in led.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r["student"], r["objective"], r["mode"], r["seed"])
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            r = dict(r); r["_shard"] = "ledger"
            runs.append(r)
            from_ledger += 1

    if not runs:
        raise SystemExit("no arms found in any blob or in the ledger; "
                         "has any worker completed an arm?")

    # Metadata comes from whichever worker actually measured it: only the forward
    # worker ran the capacity probe; the reverse workers inherited it.
    # `primary` supplies the startup metadata that is not per-arm: the coverage
    # table, the MDE, the bases.  Only a blob that actually swept arms has those, so
    # a probe-only blob must never win here.
    with_runs = [d for d in blobs.values() if d.get("runs")]
    primary = (max(with_runs, key=lambda d: len(d["runs"])) if with_runs
               else (next(iter(blobs.values())) if blobs else {}))

    # The capacity probe is chosen separately, because the worker that measured it
    # need not be the worker that swept the most arms: a worker launched with
    # --skip-probe inherits the probe from a blob named after its own tag, and when
    # that blob does not exist it records nan.  Any blob holding a probe it measured
    # itself (inherited=False) is a better source than one that inherited nothing.
    measured = [d for d in blobs.values()
                if (d.get("capacity_probe") or {}).get("per_student")
                and not (d["capacity_probe"]).get("inherited")]
    probe = (measured[0]["capacity_probe"] if measured
             else (primary.get("capacity_probe") or {}))
    students = probe.get("swept_students") or sorted({r["student"] for r in runs})
    students = [s for s in students if any(r["student"] == s for r in runs)]
    objs = sorted({r["objective"] for r in runs})
    modes = ["uniform", "early", "late", "cluster_drop"]
    modes = [m for m in modes if any(r["mode"] == m for r in runs)]
    cov = primary.get("coverage", {})

    expected = len(students) * len(objs) * len(modes) * 5
    print(f"[merge] {len(runs)} unique arms from {len(blobs)} blobs "
          f"({expected} in the full design)")
    for tag in blobs:
        print(f"           {tag}: {sum(1 for r in runs if r['_shard'] == tag)} arms kept")
    if from_ledger:
        print(f"           ledger: {from_ledger} arms kept (worker still running or "
              f"stopped before writing its blob; no per-item vectors)")
    if not blobs:
        print("[merge] no blob exists yet, so the capacity probe and the coverage "
              "table are absent and the student list is derived from the arms")
        print("[merge] NOT GATE-READY: suite_mde_points, c2_spread_ratio, base, "
              "capacity_probe and coverage are startup metadata that only a "
              "worker writes, so gate_g3 cannot read this blob.  It is for "
              "monitoring the arms that have landed; wait for one worker to "
              "finish, then merge again.")
    missing = [(st, o, m, s) for st in students for o in objs for m in modes
               for s in range(5) if (st, o, m, s) not in seen]
    if missing:
        print(f"[merge] {len(missing)} arms MISSING, e.g. "
              f"{[(slug(x[0]), x[1], x[2], x[3]) for x in missing[:4]]}")
    if a.dry_run:
        return
    if missing:
        print("[merge] writing anyway; the blob records what is missing so no "
              "reader can mistake a partial design for a complete one")

    def cell(st, obj, mode):
        return [r["accuracy"] for r in runs if r["student"] == st
                and r["objective"] == obj and r["mode"] == mode]

    anova = {}
    for st in students:
        vals = {(o, m): cell(st, o, m) for o in objs for m in modes if cell(st, o, m)}
        anova[st] = stats.eta_squared(vals) if len(vals) >= 4 else {}

    contrast = {}
    for st in students:
        for obj in objs:
            u = cell(st, obj, "uniform")
            others = [m for m in modes if m != "uniform" and cell(st, obj, m)]
            if not (u and others):
                continue
            worst = min(others, key=lambda m: np.mean(cell(st, obj, m)))
            w = cell(st, obj, worst)
            contrast[f"{slug(st)}|{obj}"] = {
                "worst_mode": worst,
                "uniform_minus_worst_points": 100 * (np.mean(u) - np.mean(w)),
                "seed_test": stats.seed_bootstrap(u, w),
                "item_test": stats.paired_bootstrap(
                    _mean_items(per_item, runs, st, obj, "uniform"),
                    _mean_items(per_item, runs, st, obj, worst))
                if _mean_items(per_item, runs, st, obj, "uniform") else {}}

    inter = {}
    for obj in objs:
        eff = [contrast.get(f"{slug(st)}|{obj}", {}).get(
            "uniform_minus_worst_points", float("nan")) for st in students]
        inter[obj] = {"students": students, "effect_points": eff,
                      "interaction_points": eff[-1] - eff[0] if len(eff) >= 2
                      else float("nan")}

    rank = stats.spearman(
        [cov.get(m, {}).get("c2", float("nan")) for m in modes],
        [np.mean([r["accuracy"] for r in runs if r["mode"] == m]) for m in modes])

    out = dict(primary)
    out.update({
        # dict(primary) copies a source worker's keys, and a worker that is still
        # running carries partial=True; the union is the experiment, so clear it.
        "partial": False,
        # Recomputed here, not copied: until 15/10 the worker reassigned the variable
        # holding the C2 spread ratio to the capacity probe's recovery spread, so a
        # blob written by the old code reports the probe number in c2_spread_ratio --
        # and nan under --skip-probe, which would fail validity condition V4.6 on the
        # reverse workers' blobs.  The coverage table is the measurement itself, so
        # derive the ratio from it and the worker's field never matters.
        "c2_spread_ratio": _c2_spread(cov, modes),
        "capacity_probe": probe,
        "capacity_probe_source": next((k for k, d in blobs.items()
                                       if d.get("capacity_probe") is probe), None),
        "c2_spread_ratio_reported_by_worker": primary.get("c2_spread_ratio"),
        "merged_from": sorted(blobs), "arms_from_ledger": from_ledger,
        "merged_from_partial": sorted(k for k, d in blobs.items()
                                      if d.get("partial")),
        "runs": runs, "per_item": per_item,
        "n_arms": len(runs), "n_expected": expected,
        "missing_arms": [[x[0], x[1], x[2], x[3]] for x in missing],
        "anova": anova, "contrast": contrast, "interaction": inter,
        "rank_c2_vs_accuracy": rank,
        "reliability": stats.reliability(
            [cell(st, o, m) for st in students for o in objs for m in modes
             if len(cell(st, o, m)) >= 2]),
        "oom_arms": _dedup_oom(blobs, completed=seen),
        "oom_arms_later_completed": sum(
            1 for d in blobs.values() for o in (d.get("oom_arms") or [])
            if (o.get("student"), o.get("objective"), o.get("mode"),
                o.get("seed")) in seen),
    })
    runio.save(a.out, out)

    print(f"\n{'student':>14} {'obj':>4} " + " ".join(f"{m:>13}" for m in modes))
    for st in students:
        for o in objs:
            row = " ".join(
                f"{100*np.mean(cell(st, o, m)):13.2f}" if cell(st, o, m) else f"{'-':>13}"
                for m in modes)
            print(f"{slug(st):>14} {o:>4} {row}")
    print()
    for st in students:
        e = anova.get(st) or {}
        if e:
            print(f"  {slug(st):>14}: eta2 measure={e['eta2_measure']:.3f} "
                  f"loss={e['eta2_loss']:.3f} "
                  f"ratio={e['eta2_measure']/max(e['eta2_loss'],1e-9):.2f}")
    print(f"  rank C2 vs accuracy: rho={rank['rho']:+.3f}")
    for o, v in inter.items():
        print(f"  interaction [{o}]: effects {[round(x,2) for x in v['effect_points']]}"
              f" -> {v['interaction_points']:+.2f} pts")


if __name__ == "__main__":
    main()
