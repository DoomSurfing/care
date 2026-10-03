# CARE-style MoE-LoRA research (Qwen2.5-7B, commonsense-170k)

Goal: study confidence-adaptive expert routing for MoE-LoRA, using CARE as a reference point.

## Layout
- `src/`     library code (data, MoE-LoRA layer, gates, prompts, config)
- `scripts/` numbered pipeline scripts (00 env check ... 08 aggregate) and `run_queue.sh`
- `tests/`   CPU tests and data tests (run before any real experiment)
- `results/` evaluation JSON outputs
- `logs/`    terminal logs of setup and evaluation steps

## Reproduce
    conda activate care && export PYTHONPATH=src
    python scripts/01_download_data.py
    PYTHONPATH=src python scripts/02_preflight.py
    SEEDS="42" bash scripts/run_queue.sh moe_fixed4

## Labeled assumptions (not given by the paper)
See `ASSUMPTIONS` in `src/common.py` (lora_alpha=16, aux_coef=0.01, weight_decay=0, no grad clip, no warmup,
train max_len=256 with left-truncation, 2000-example holdout, loss on answer+EOS only).

## Hardware
Single H200 MIG partition (~69.75 GiB usable). Batch 16, no gradient checkpointing, no accumulation.
