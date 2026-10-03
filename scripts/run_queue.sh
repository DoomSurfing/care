#!/usr/bin/env bash
# usage: SEEDS="42 43 44" bash scripts/run_queue.sh moe_fixed4
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTORCH_ALLOC_CONF=expandable_segments:True
METHOD=${1:-moe_fixed4}
SEEDS=${SEEDS:-"42"}
mkdir -p logs/train
for s in $SEEDS; do
  if [ -f "ckpt/${METHOD}_s${s}/final.pt" ]; then echo "[skip] ${METHOD}_s${s} done"; continue; fi
  echo "=== ${METHOD} seed ${s} $(date) ==="
  python scripts/06_train.py --method "$METHOD" --seed "$s" --resume --wandb 2>&1 | tee -a "logs/train/${METHOD}_s${s}.log" \
    || { echo "FAILED seed ${s}"; exit 1; }
done
echo "QUEUE DONE $(date)"