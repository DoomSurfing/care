#!/usr/bin/env bash
# RECIPE B: AdamW betas (0.9,0.95), warmup 3%, clip 1.0, SpikeGuard OFF (non-finite steps only), watchdog OFF.
# Phase 1: per seed train -> eval fixed k=4 (LL+gen); the FIRST seed is health-gated.  Phase 2: fixed-k sweep (LL) on all seeds.
# Resumable: finished work is skipped, so re-running this script continues. Never rm ckpt/ from here.
cd /workspace/projects/care || exit 1
source /root/miniconda3/etc/profile.d/conda.sh && conda activate care || { echo "conda activate failed"; exit 1; }
export PYTHONPATH=src PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs/overnight logs/train results/overnight_summary results_backup
exec 9>/tmp/care_overnight.lock
flock -n 9 || { echo "another run holds the lock"; exit 1; }
exec > >(tee -a "logs/overnight/recipeB_$(date +%Y%m%d_%H%M%S).log") 2>&1

TRAIN_SEEDS=${TRAIN_SEEDS:-"44 43 46 45 42"}
ALL_SEEDS=${ALL_SEEDS:-"42 43 44 45 46"}
SWEEP_KS=${SWEEP_KS:-"1 2 3 6 8"}
STATUS=logs/overnight/status.tsv
rm -f STOP_OVERNIGHT logs/overnight/DONE

note() { echo -e "$(date '+%F %T')\t$*" | tee -a "$STATUS"; }
stop_requested() { if [ -f STOP_OVERNIGHT ]; then note "STOP file found; stopping"; return 0; fi; return 1; }
collect() { python scripts/17_collect.py > logs/overnight/last_summary.txt 2>&1 || note "COLLECT_FAILED"; }
git_snapshot() {
  git add results/overnight_summary results/moe_fixed4_s* 2>/dev/null
  git commit -q -m "exp: recipeB results ($1)" 2>/dev/null && timeout 120 git push -q 2>/dev/null
  return 0
}
backup_results() {
  local run=$1
  mkdir -p "results_backup/$run" "results/$run/train_logs"
  cp -a "results/$run/." "results_backup/$run/" 2>/dev/null
  for f in "logs/train/$run.jsonl" "logs/train/$run.val.jsonl" "logs/train/$run.skips.jsonl"; do cp "$f" "results/$run/train_logs/" 2>/dev/null; done
  cp "ckpt/$run/train_summary.json" "results/$run/" 2>/dev/null
  return 0
}

gate_check() {
  python - "$1" <<'PY'
import json, sys
s = sys.argv[1]
rows = [json.loads(l) for l in open(f"logs/train/moe_fixed4_s{s}.val.jsonl") if l.strip()]
last = rows[-1]
ok = last["val_loss"] <= 0.25 and last["val_exact"] >= 90
print(f"GATE s{s}: final val_loss {last['val_loss']:.3f} exact {last['val_exact']:.1f} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
}

train_seed() {
  local s=$1 run="moe_fixed4_s$1" rc attempt avail
  if [ -f "ckpt/$run/final.pt" ]; then note "TRAIN_SKIP $run (final.pt exists)"; return 0; fi
  for attempt in 1 2 3; do
    avail=$(df --output=avail -BG /workspace 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ "${avail:-999}" -lt 22 ]; then note "TRAIN_ABORT $run: only ${avail}G free (<22G)"; return 1; fi
    note "TRAIN_START $run attempt $attempt (recipe B)"
    python scripts/06_train.py --method moe_fixed4 --seed "$s" --resume --skip_mult 0.0 --no_watchdog --wandb 2>&1 | tee -a "logs/train/$run.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ] && [ -f "ckpt/$run/final.pt" ]; then
      note "TRAIN_DONE $run | last val: $(tail -n 1 logs/train/$run.val.jsonl 2>/dev/null)"
      return 0
    fi
    note "TRAIN_CRASH $run rc=$rc"; sleep 60
  done
  return 1
}

eval_one() {
  local s=$1 tag=$2 attempt rc; shift 2
  local run="moe_fixed4_s$s" out="results/moe_fixed4_s$s/$tag.json"
  if [ -f "$out" ]; then note "EVAL_SKIP $run $tag"; return 0; fi
  if [ ! -f "ckpt/$run/final.pt" ]; then note "EVAL_NOFINAL $run $tag"; return 1; fi
  for attempt in 1 2; do
    note "EVAL_START $run $tag attempt $attempt"
    python scripts/07_eval.py --run "$run" "$@" 2>&1 | tee -a "logs/overnight/eval_${run}_${tag}.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ] && [ -f "$out" ]; then
      note "EVAL_DONE $run $tag | $(grep AVERAGE logs/overnight/eval_${run}_${tag}.log | tail -n 1)"
      backup_results "$run"; return 0
    fi
    note "EVAL_FAIL $run $tag rc=$rc"; sleep 30
  done
  return 1
}

note "RECIPEB_START train=[$TRAIN_SEEDS] all=[$ALL_SEEDS] ks=[$SWEEP_KS] | guard OFF, watchdog OFF, beta2 0.95, warmup 3%, clip 1.0"
first=1
for s in $TRAIN_SEEDS; do
  stop_requested && break
  train_seed "$s"; trc=$?
  if [ "$trc" -eq 0 ]; then
    stop_requested && break
    eval_one "$s" fixed4 --gate fixed --k 4 --collect_k
    if [ "$first" -eq 1 ]; then
      gate_out=$(gate_check "$s"); grc=$?
      note "$gate_out"; first=0
      if [ "$grc" -ne 0 ]; then
        note "GATE_FAIL: stopping before further seeds; inspect and decide"
        collect; git_snapshot "gate fail s$s"; touch logs/overnight/DONE; exit 3
      fi
    fi
  else
    note "TRAIN_NOT_OK moe_fixed4_s$s rc=$trc"
  fi
  collect; git_snapshot "seed $s"; sleep 15
done

for s in $ALL_SEEDS; do
  for K in $SWEEP_KS; do
    stop_requested && break 2
    eval_one "$s" "fixed$K" --gate fixed --k "$K" --skip_gen
  done
  collect; git_snapshot "sweep seed $s"
done
collect; git_snapshot "final"
note "RECIPEB_DONE"; touch logs/overnight/DONE
