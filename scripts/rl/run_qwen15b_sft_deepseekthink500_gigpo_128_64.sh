#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR=${MODEL_DIR:-/root/autodl-fs/WarmGiGPO-WebShop/outputs/sft/qwen25_1p5b_webshop_lora_verl_deepseek_think_500/merged_model}
export TRAIN_FILE=${TRAIN_FILE:-/root/data/verl-agent/text_128_64/train.parquet}
export VAL_FILE=${VAL_FILE:-/root/data/verl-agent/text_128_64/test.parquet}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen15b_sft_deepseekthink500_gigpo_128_64}

bash scripts/rl/run_qwen15b_sft_verl_gigpo_medium.sh
