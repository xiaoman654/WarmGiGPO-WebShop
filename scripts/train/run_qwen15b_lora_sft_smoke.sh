#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent-webshop
source /etc/network_turbo || true

export OMP_NUM_THREADS=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_DIR=${MODEL_DIR:-/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WarmGiGPO-WebShop}

cd "$PROJECT_DIR"

python scripts/train/sft_lora.py \
  --model-name-or-path "$MODEL_DIR" \
  --train-file data/processed/sft_step_level/train.jsonl \
  --valid-file data/processed/sft_step_level/valid.jsonl \
  --output-dir outputs/sft/qwen25_1p5b_webshop_lora_smoke \
  --max-train-samples 500 \
  --max-valid-samples 100 \
  --max-seq-length 2048 \
  --num-train-epochs 1 \
  --max-steps 50 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --logging-steps 5 \
  --eval-steps 25 \
  --save-steps 25 \
  --save-total-limit 2 \
  --bf16 \
  --gradient-checkpointing

