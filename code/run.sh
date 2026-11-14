#!/usr/bin/env bash
# =============================================================================
#  State-Coverage Distillation -- one stage at a time, never the whole study.
#
#    bash run.sh info          print the configuration, run nothing
#    bash run.sh gp            stage GP  (positive control)   ~6 GPU-h
#    bash run.sh g0            stage G0  (estimators, CPU)    ~1 day CPU
#    bash run.sh g1            stage G1  (theory, CPU)        ~1 h  CPU
#    bash run.sh g2            stage G2  (validity)           ~8 GPU-h
#    bash run.sh g6            stage G6  (mechanism)          ~2 GPU-h   <- before g3
#    bash run.sh g3            stage G3  (measure vs loss)    ~25 GPU-h
#    bash run.sh g4            stage G4  (prediction grid)    ~130 GPU-h
#    bash run.sh g5            stage G5  (main comparison)    ~250 GPU-h
#    bash run.sh g7            stage G7  (dissociation)       ~46 GPU-h
#    bash run.sh g8            stage G8  (cost, pass@k)       ~30 GPU-h
#    bash run.sh gate <stage>  re-print a verdict without running anything
#    bash run.sh selftest      test the gate harness itself, run nothing
#    bash run.sh deps          check the imports every stage needs
#
#  SCD_OVERRIDE_<STAGE>="reason" bypasses ONE prerequisite gate, loudly and on the
#  record.  Used once: G6 refuted Proposition 2, and the registered consequence was
#  to stop -- but only the budget-aiming failed, and G3 never uses it.
#    bash run.sh status        what exists, what is missing
#
#  A stage whose result file exists and whose gate already passes is SKIPPED, so
#  resubmitting a chain after a fix does not repeat the work before it.  FORCE=1
#  reruns regardless.
#
#  There is deliberately NO "run everything" target.  Each stage ends by printing
#  a gate verdict; the next stage is a decision, not a continuation.  A stage
#  whose prerequisite gate has not passed refuses to start unless FORCE=1.
#
#  Results land in results/ when file names cannot collide across stages, and in
#  results-<x>/ for the stages that write one file per grid point.
#
#  Override anything from the environment:
#    TEACHER=Qwen/Qwen2.5-14B-Instruct-AWQ GEN_BATCH=8 bash run.sh gp
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-/workspace/miniconda3/bin/python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export SCD_VRAM_GB=${SCD_VRAM_GB:-30}

# ------------------------------------------------------------- configuration
TEACHER=${TEACHER:-Qwen/Qwen2.5-7B-Instruct}
TEACHER_BIG=${TEACHER_BIG:-Qwen/Qwen2.5-14B-Instruct-AWQ}
# Students are the BASE checkpoints, not the -Instruct ones.  The first GP run used
# Qwen2.5-0.5B-Instruct, which already scored 41.3% on the pooled suite and left
# only 3.34 points of headroom -- about a tenth of the teacher-minus-student gap,
# and 1.6x the design's minimum detectable effect against a required 4x.  A base
# student is both the setting the surveys describe (moving capability into a cheap
# artefact, not sharpening one that is already tuned) and the one with room to
# measure in.  GP guards the obvious objection by scoring the base model few-shot
# as well as zero-shot and taking the better of the two as its reference.
STUDENT_S=${STUDENT_S:-Qwen/Qwen2.5-0.5B}
STUDENT_L=${STUDENT_L:-Qwen/Qwen2.5-1.5B}
# A third candidate for G3's capacity probe.  GP measured the 0.5B student
# recovering 47.5% of a 65.6-point gap -- the capacity-limited end, where G1 says
# coverage does not matter.  Probing a 3B as well costs one extra arm and removes
# the risk that the sweep lands entirely in the wrong regime.
STUDENT_XL=${STUDENT_XL:-Qwen/Qwen2.5-3B}

PROMPTS=${PROMPTS:-6000}          # distillation prompts
SAMPLES=${SAMPLES:-4}             # teacher traces per prompt
FRAC=${FRAC:-0.25}                # supervised share of steps in G3/G4
BRANCH=${BRANCH:-4000}            # branch continuations per SCD round
EPOCHS=${EPOCHS:-3}
LORA_R=${LORA_R:-32}
LR=${LR:-1e-4}
GP_LRS=${GP_LRS:-"2e-5 5e-5 1e-4 2e-4"}

# 30 GB VRAM: these are the only knobs that matter.  A 7B teacher in bfloat16 is
# 15.2 GB, so generation gets a small batch; a 1.5B student with adapters trains
# comfortably at batch 4 x accum 4.
GEN_BATCH=${GEN_BATCH:-16}
# TRAIN_BATCH x ACCUM is the effective batch and must stay 16 across every arm.
# If TRAIN_BATCH has to come down for memory, raise ACCUM by the same factor --
# otherwise the arm is no longer comparable with the arms already run.
TRAIN_BATCH=${TRAIN_BATCH:-4}
ACCUM=${ACCUM:-4}
# 1 trades about a third of the step time for roughly a tenth of the activation
# memory.  Reach for it only when the guard says activations are binding: the
# logits are no longer the problem (see scd/losses.py).
GRAD_CHECKPOINTING=${GRAD_CHECKPOINTING:-0}
# G6 sweeps the branch budget.  It must stay well under the coupon-collector
# threshold n*ln(n) for the number of cells the teacher visits -- at 413 cells that
# is 2488 draws, and a single budget of 4000 saturated random selection so
# completely that every rule landed within 0.4 of the same answer.
G6_BUDGETS=${G6_BUDGETS:-"100 200 400 800"}
export SCD_GRAD_CHECKPOINTING=$GRAD_CHECKPOINTING
EVAL_BATCH=${EVAL_BATCH:-16}
LOGIT_BATCH=${LOGIT_BATCH:-4}
HIDDEN_BATCH=${HIDDEN_BATCH:-8}

SEEDS=${SEEDS:-"0 1 2 3 4"}
SEEDS3=${SEEDS3:-"0 1 2"}
SUITE_LIMIT=${SUITE_LIMIT:-0}     # >0 shrinks the suite; for smoke tests only

mkdir -p results results-syn results-g4 results-g5 results-g7 logs artifacts figs \
         latex/generated

# Where each stage's verdict reads from.  One table, used by run_stage to skip
# finished work and by `status` to report it -- two copies would drift.
result_file () {
  case "$1" in
    gp) echo results/gp_control.json ;;
    g0) echo results-syn/g0_estimator.json ;;
    g1) echo results-syn/g1_theory.json ;;
    g2) echo results/g2_validity.json ;;
    g6) echo results/g6_mechanism.json ;;
    g3) echo results/g3_measure_vs_loss.json ;;
    g4) echo results-g4/g4_predict.json ;;
    g5) echo results-g5/g5_main.json ;;
    g7) echo results-g7/g7_dissociation.json ;;
    g8) echo results/g8_cost.json ;;
  esac
}

say () { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*" | tee -a logs/run.log; }

# ------------------------------------------------------------- gate machinery
gate () {   # gate <stage>
  "$PY" -c "from scd import gates; import sys; sys.exit(gates.run('$1'))"
}

# The verdict predicate has been wrong twice, in opposite directions, and the
# second time it printed FAIL over six passing criteria and stopped a chain that
# had just produced a 31-point gain.  It is now tested before every stage: this
# costs milliseconds and sits in front of every GPU hour the study spends.
# requirements.txt listing a package is not the same as the server having it.
# G0 died on a missing scikit-learn, and a chain run would have reached that
# failure only after clearing GP -- so the imports a stage needs are checked
# before it starts, in milliseconds.
#
# Per stage, not globally: G0 and G1 are CPU-only and need nothing from torch, so
# a global check would block the two cheapest gates on a machine that cannot run
# the expensive ones.  That is the opposite of what the ladder is for.
deps_check () {   # deps_check [stage]
  # The heredoc must not sit inside a `|| { ... }` group: bash starts reading the
  # heredoc body after the current line and the brace group then fails to parse.
  # Capture the status, then branch.
  "$PY" - "${1:-all}" <<'PYCHK'
import importlib.util, sys
stage = sys.argv[1] if len(sys.argv) > 1 else "all"
cpu_only = {"numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn"}
gpu = {"torch": "torch", "transformers": "transformers",
       "datasets": "datasets", "peft": "peft"}
need = dict(cpu_only) if stage in ("g0", "g1") else {**cpu_only, **gpu}
missing = [pkg for mod, pkg in need.items() if not importlib.util.find_spec(mod)]
if missing:
    print(f"[deps] stage {stage} needs: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PYCHK
  local st=$?
  if [ "$st" -ne 0 ]; then
    printf '\033[31m[abort]\033[0m missing dependencies; run: %s -m pip install -r requirements.txt\n' "$PY"
    return 1
  fi
  return 0
}

gate_selftest () {
  "$PY" -c "from scd import gates; gates.selftest()" || {
    printf '\033[31m[abort]\033[0m the gate harness failed its own self-test.\n'
    return 1
  }
}

# An explicit, loud, recorded override of a prerequisite gate.
#
# G6 refuted Proposition 2: no selection rule beat random branching by more than
# 1.0235x against a 1.5x bar, and at the scarcest budget random was the best rule.
# The registered consequence was to stop the direction. But that sentence was
# written for a direction in which measuring coverage and aiming a budget stood or
# fell together, and only the aiming failed -- Theorem 1 (0/780 violations),
# Corollary 1.1 (47 configurations) and estimator A (validated against ground
# truth) are about measurement. G3 manipulates mu by masking and never uses
# Proposition 2.
#
# Narrowing a registered stop condition is exactly the goalpost-moving this
# protocol exists to prevent, so it is not done quietly: the override needs a
# reason, prints a banner every run, and lands in the result blob through the
# environment that runio records. A reader can see it; a future run cannot forget
# it.
override_reason () {  # override_reason <stage>
  local v="SCD_OVERRIDE_$(echo "$1" | tr "[:lower:]" "[:upper:]")"
  eval "printf '%s' \"\${$v:-}\""
}

require () {  # require <stage> ... -- refuse to start until these have passed
  [ "${FORCE:-0}" = "1" ] && { say "FORCE=1: skipping prerequisite checks"; return 0; }
  local bad=0
  for g in "$@"; do
    local why; why=$(override_reason "$g")
    if [ -n "$why" ]; then
      printf '\033[33m[OVERRIDE]\033[0m prerequisite gate %s bypassed deliberately.\n' "$g"
      printf '           reason: %s\n' "$why"
      printf '           A recorded deviation from a pre-registered stop condition.\n'
      continue
    fi
    if ! gate "$g" >/dev/null 2>&1; then
      printf '\033[31m[blocked]\033[0m prerequisite gate %s has not passed.\n' "$g"
      gate "$g" || true
      bad=1
    fi
  done
  if [ "$bad" = 1 ]; then
    printf '\nRefusing to run: fix the prerequisite, or set FORCE=1 if you\n'
    printf 'deliberately want an out-of-order run (and say so in the paper).\n'
    return 1
  fi
}

run_stage () {  # run_stage <name> <results-dir> <script> [args...]
  local name=$1 outdir=$2; shift 2
  deps_check "$name" || return 1
  gate_selftest || return 1

  # Finished work is re-gated, not re-run.  Without this, resubmitting a chain
  # after fixing a later stage repeats everything before it -- four hours for GP
  # alone -- and the whole point of writing one result file per stage is that it
  # does not have to.
  local rf; rf=$(result_file "$name")
  if [ "${FORCE:-0}" != "1" ] && [ -n "$rf" ] && [ -f "$rf" ]; then
    if SCD_RESULTS="$outdir" gate "$name" >/dev/null 2>&1; then
      say "stage $name already done and its gate passes -- skipping"
      SCD_RESULTS="$outdir" gate "$name"
      return 0
    fi
    say "stage $name has a result file but its gate does not pass; rerunning"
  fi

  say "stage $name  ->  $outdir/"
  SCD_RESULTS="$outdir" "$PY" -u "$@" 2>&1 \
    | tee -a "logs/$name.log" \
    | grep --line-buffered -vE 'it/s\]|examples/s\]|^Loading weights'
  # --line-buffered is not cosmetic: without it grep holds the progress lines in
  # its own buffer, so a stage running normally looks hung.  A G0 run showed only
  # the stage header for fifty-six minutes in the redirected output while
  # logs/g0.log, written by tee before the pipe, already read 160/780.
  local st=${PIPESTATUS[0]}
  if [ "$st" -ne 0 ]; then say "stage $name FAILED (exit $st)"; return "$st"; fi
  say "stage $name finished; verdict follows"
  SCD_RESULTS="$outdir" gate "$name"
  return $?
}

# ------------------------------------------------------------------- stages
case "${1:-help}" in

info)
  echo "python        : $PY"
  echo "teacher       : $TEACHER   (big: $TEACHER_BIG)"
  echo "students      : $STUDENT_S , $STUDENT_L"
  echo "prompts x n   : $PROMPTS x $SAMPLES"
  echo "vram budget   : ${SCD_VRAM_GB} GB"
  echo "batches       : gen=$GEN_BATCH train=$TRAIN_BATCH x $ACCUM "\
       "(effective $((TRAIN_BATCH*ACCUM))) eval=$EVAL_BATCH"
  echo "grad ckpt     : $GRAD_CHECKPOINTING"
  echo "seeds         : $SEEDS  (3-seed stages: $SEEDS3)"
  "$PY" -c "import torch;print('cuda      :',torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null || true
  ;;

gp)
  run_stage gp results experiments/gp_control.py \
    --teacher "$TEACHER" --student "$STUDENT_S" \
    --prompts "$PROMPTS" --samples "$SAMPLES" --epochs "$EPOCHS" \
    --lora-r "$LORA_R" --batch "$TRAIN_BATCH" --accum "$ACCUM" \
    --gen-batch "$GEN_BATCH" --eval-batch "$EVAL_BATCH" \
    --seeds "$SEEDS3" --suite-limit "$SUITE_LIMIT" --lrs "$GP_LRS"
  ;;

g0)
  require gp || exit 1
  run_stage g0 results-syn experiments/g0_estimator.py \
    --worlds "${WORLDS:-12}" --ns "${NS:-1000 10000 100000}" --reps "${REPS:-3}"
  ;;

g1)
  require gp g0 || exit 1
  run_stage g1 results-syn experiments/g1_theory.py --worlds "${WORLDS:-12}"
  ;;

g2)
  require gp g0 g1 || exit 1
  run_stage g2 results experiments/g2_validity.py \
    --teacher "$TEACHER" --student "$STUDENT_S" \
    --prompts "${G2_PROMPTS:-2500}" --samples "$SAMPLES" \
    --gen-batch "$GEN_BATCH" --hidden-batch "$HIDDEN_BATCH"
  ;;

g6)
  require gp g0 g1 g2 || exit 1
  run_stage g6 results experiments/g6_mechanism.py \
    --teacher "$TEACHER" --prompts "$PROMPTS" --samples "$SAMPLES" \
    --budgets "$G6_BUDGETS" --gen-batch "$GEN_BATCH"
  ;;

g3)
  require gp g0 g1 g2 g6 || exit 1
  run_stage g3 results experiments/g3_measure_vs_loss.py \
    --teacher "$TEACHER" --students "${G3_STUDENTS:-$STUDENT_S $STUDENT_L $STUDENT_XL}" \
    --sweep-students "${G3_SWEEP:-2}" \
    --prompts "$PROMPTS" --samples "$SAMPLES" --frac "$FRAC" \
    --seeds "$SEEDS" --epochs "$EPOCHS" --lora-r "$LORA_R" \
    --batch "$TRAIN_BATCH" --accum "$ACCUM" --gen-batch "$GEN_BATCH" \
    --eval-batch "$EVAL_BATCH" --logit-batch "$LOGIT_BATCH" \
    --suite-limit "$SUITE_LIMIT"
  ;;

g4)
  require g3 || exit 1
  run_stage g4 results-g4 experiments/g4_predict.py \
    --prompts "${G4_PROMPTS:-4000}" --samples "$SAMPLES" --frac "$FRAC" \
    --seeds "$SEEDS3" --epochs "$EPOCHS" --lora-r "$LORA_R" \
    --batch "$TRAIN_BATCH" --accum "$ACCUM" --gen-batch "$GEN_BATCH" \
    --eval-batch "$EVAL_BATCH" --logit-batch "$LOGIT_BATCH" --lr "$LR" \
    --suite-limit "$SUITE_LIMIT"
  ;;

g5)
  require g3 g6 || exit 1
  run_stage g5 results-g5 experiments/g5_main.py \
    --teachers "$TEACHER $TEACHER_BIG" --students "$STUDENT_S $STUDENT_L" \
    --prompts "$PROMPTS" --samples "$SAMPLES" --branch-budget "$BRANCH" \
    --seeds "$SEEDS" --epochs "$EPOCHS" --lora-r "$LORA_R" --lr "$LR" \
    --batch "$TRAIN_BATCH" --accum "$ACCUM" --gen-batch "$GEN_BATCH" \
    --eval-batch "$EVAL_BATCH" --logit-batch "$LOGIT_BATCH" \
    --suite-limit "$SUITE_LIMIT" --passk "${PASSK:-0}"
  ;;

g7)
  require g3 || exit 1
  run_stage g7 results-g7 experiments/g7_dissociation.py \
    --teacher "$TEACHER" --student "$STUDENT_L" \
    --seeds "$SEEDS3" --epochs "$EPOCHS" --lora-r "$LORA_R" --lr "$LR" \
    --batch "$TRAIN_BATCH" --accum "$ACCUM" --gen-batch "$GEN_BATCH" \
    --eval-batch "$EVAL_BATCH" --suite-limit "$SUITE_LIMIT"
  ;;

g8)
  require g5 || exit 1
  run_stage g8 results experiments/g8_cost.py \
    --teacher "$TEACHER" --student "$STUDENT_L" \
    --prompts "$PROMPTS" --samples "$SAMPLES" --branch-budget "$BRANCH" \
    --seeds "$SEEDS3" --epochs "$EPOCHS" --lora-r "$LORA_R" --lr "$LR" \
    --batch "$TRAIN_BATCH" --accum "$ACCUM" --gen-batch "$GEN_BATCH" \
    --eval-batch "$EVAL_BATCH"
  ;;

smoke)
  say "smoke: tiny end-to-end run, results go to results-smoke/ and prove nothing"
  mkdir -p results-smoke
  SUITE_LIMIT=40 PROMPTS=60 SAMPLES=2 SEEDS3="0" \
  SCD_RESULTS=results-smoke "$PY" -u experiments/gp_control.py \
    --teacher "$TEACHER" --student "$STUDENT_S" --prompts 60 --samples 2 \
    --lrs "1e-4" --seeds "0" --epochs 1 --suite-limit 40 \
    --gen-batch "$GEN_BATCH" --eval-batch "$EVAL_BATCH" --tag smoke_gp
  SCD_RESULTS=results-smoke "$PY" -u experiments/g0_estimator.py \
    --worlds 2 --ns "1000" --reps 1 --tag smoke_g0
  SCD_RESULTS=results-smoke "$PY" -u experiments/g1_theory.py \
    --worlds 2 --tag smoke_g1
  # G2, G6 and G3 are the expensive ones (8, 2 and 25 GPU-hours), so their code
  # paths are exercised here too.  A crash found in fifteen minutes beats the same
  # crash found after eight hours; the numbers these produce are meaningless and
  # land in results-smoke/, which no gate and no table ever reads.
  SCD_RESULTS=results-smoke "$PY" -u experiments/g2_validity.py \
    --teacher "$TEACHER" --student "$STUDENT_S" --prompts 40 --samples 2 \
    --limit 200 --gen-batch "$GEN_BATCH" --hidden-batch "$HIDDEN_BATCH" \
    --mag-bins "24" --tag smoke_g2
  SCD_RESULTS=results-smoke "$PY" -u experiments/g6_mechanism.py \
    --teacher "$TEACHER" --prompts 40 --samples 2 --budget 40 \
    --gen-batch "$GEN_BATCH" --tag smoke_g6
  SCD_RESULTS=results-smoke "$PY" -u experiments/g3_measure_vs_loss.py \
    --teacher "$TEACHER" --students "$STUDENT_S" --prompts 40 --samples 2 \
    --seeds "0" --epochs 1 --objectives "ce fkl" --suite-limit 40 \
    --batch "$TRAIN_BATCH" --accum "$ACCUM" --gen-batch "$GEN_BATCH" \
    --eval-batch "$EVAL_BATCH" --logit-batch "$LOGIT_BATCH" \
    --force-full 1 --tag smoke_g3
  say "smoke finished. Delete results-smoke/ before a real run."
  ;;

selftest)
  "$PY" -c "from scd import gates; gates.selftest()"
  ;;

deps)
  deps_check "${2:-all}" && echo "[deps] ${2:-all}: all present"
  ;;

gate)
  [ $# -ge 2 ] || { echo "usage: bash run.sh gate <stage>"; exit 2; }
  case "$2" in
    g0|g1) SCD_RESULTS=results-syn gate "$2" ;;
    g4)    SCD_RESULTS=results-g4  gate g4 ;;
    g5)    SCD_RESULTS=results-g5  gate g5 ;;
    g7)    SCD_RESULTS=results-g7  gate g7 ;;
    *)     SCD_RESULTS=results     gate "$2" ;;
  esac
  ;;

status)
  printf '%-6s %-28s %s\n' STAGE FILE STATUS
  for s in gp g0 g1 g2 g6 g3 g4 g5 g7 g8; do
    set -- "$s" "$(result_file "$s")"
    if [ -f "$2" ]; then
      printf '%-6s %-28s \033[32mdone\033[0m  (%s)\n' "$1" "$2" \
        "$(date -r "$2" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '')"
    else
      printf '%-6s %-28s \033[33mmissing\033[0m\n' "$1" "$2"
    fi
  done
  echo
  du -sh artifacts 2>/dev/null | sed 's/^/cached artefacts: /'
  ;;

tables)   "$PY" -u make_tables.py ;;
figures)  "$PY" -u make_figures.py ;;

*)
  sed -n '2,30p' "$0"
  ;;
esac
