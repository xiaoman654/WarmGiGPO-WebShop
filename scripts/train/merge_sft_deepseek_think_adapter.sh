#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent-webshop
source /etc/network_turbo || true

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

MODEL_DIR=${MODEL_DIR:-/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WarmGiGPO-WebShop}

cd "$PROJECT_DIR"

python scripts/train/merge_lora_adapter.py \
  --base-model "$MODEL_DIR" \
  --adapter outputs/sft/qwen25_1p5b_webshop_lora_verl_deepseek_think_500/final_adapter \
  --output-dir outputs/sft/qwen25_1p5b_webshop_lora_verl_deepseek_think_500/merged_model
