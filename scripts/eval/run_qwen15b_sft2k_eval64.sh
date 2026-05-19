#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR=${MODEL_DIR:-/root/autodl-fs/WarmGiGPO-WebShop/outputs/sft/qwen25_1p5b_webshop_lora_verl_2k/merged_model}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen15b_sft2k_eval64}

bash scripts/eval/run_qwen15b_sft_verl_full_eval64.sh
