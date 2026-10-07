#!/usr/bin/env bash
cd /workspace/projects/care || exit 1
source /root/miniconda3/etc/profile.d/conda.sh && conda activate care || exit 1
export PYTHONPATH=src PYTORCH_ALLOC_CONF=expandable_segments:True

until [ -f logs/overnight/DONE ]; do sleep 60; done

for pair in "43 tripped_25887" "45 tripped_22583" "46 tripped_10880"; do
  set -- $pair; s=$1; t=$2
  snap=$(ls ckpt/moe_fixed4_s$s/snap_*.pt 2>/dev/null | sed 's#.*/##; s#\.pt$##' | sort -t_ -k2 -n | tail -1)
  echo "== seed $s: $t vs ${snap:-<no snapshot>}"
  python scripts/14_val_loss.py --run moe_fixed4_s$s --ckpts $t $snap 2>&1 | grep -E "^(tripped|snap)_"
done
echo "== done"
