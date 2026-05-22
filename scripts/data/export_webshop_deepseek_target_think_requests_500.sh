#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WarmGiGPO-WebShop}

cd "$PROJECT_DIR"

if [ ! -f data/processed/sft_step_level_verl_multiturn/train.jsonl ]; then
  bash scripts/data/build_webshop_sft_multiturn_verl.sh
fi

mkdir -p data/processed/deepseek_target_think_requests

python scripts/data/export_webshop_think_requests.py \
  --input data/processed/sft_step_level_verl_multiturn/train.jsonl \
  --output data/processed/deepseek_target_think_requests/webshop_multiturn_train500_requests.jsonl \
  --max-samples 500 \
  --max-words 60 \
  --include-target-action

python scripts/data/export_webshop_think_requests.py \
  --input data/processed/sft_step_level_verl_multiturn/valid.jsonl \
  --output data/processed/deepseek_target_think_requests/webshop_multiturn_valid500_requests.jsonl \
  --max-samples 500 \
  --max-words 60 \
  --include-target-action

cat \
  data/processed/deepseek_target_think_requests/webshop_multiturn_train500_requests.jsonl \
  data/processed/deepseek_target_think_requests/webshop_multiturn_valid500_requests.jsonl \
  > data/processed/deepseek_target_think_requests/webshop_multiturn_500_requests.jsonl

echo "wrote combined requests: data/processed/deepseek_target_think_requests/webshop_multiturn_500_requests.jsonl"
