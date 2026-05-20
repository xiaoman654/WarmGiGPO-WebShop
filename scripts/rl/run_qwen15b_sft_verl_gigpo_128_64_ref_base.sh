#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR=${MODEL_DIR:-/root/autodl-fs/WarmGiGPO-WebShop/outputs/sft/qwen25_1p5b_webshop_lora_verl_full/merged_model}
export REF_MODEL_DIR=${REF_MODEL_DIR:-/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
export TRAIN_FILE=${TRAIN_FILE:-/root/data/verl-agent/text_128_64/train.parquet}
export VAL_FILE=${VAL_FILE:-/root/data/verl-agent/text_128_64/test.parquet}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen15b_sft_verl_gigpo_128_64_ref_base}

bash scripts/rl/apply_verl_ref_model_path_patch.sh
bash scripts/rl/run_qwen15b_sft_verl_gigpo_medium.sh
