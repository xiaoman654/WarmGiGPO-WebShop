#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WarmGiGPO-WebShop}
THINK_FILE=${THINK_FILE:-data/processed/deepseek_think_requests/webshop_multiturn_500_responses.jsonl}

cd "$PROJECT_DIR"

if [ ! -f "$THINK_FILE" ]; then
  echo "Missing THINK_FILE=$THINK_FILE" >&2
  echo "Generate it from data/processed/deepseek_think_requests/webshop_multiturn_train500_requests.jsonl first." >&2
  exit 2
fi

python scripts/data/build_webshop_sft_with_generated_think.py \
  --base-train data/processed/sft_step_level_verl_multiturn/train.jsonl \
  --base-valid data/processed/sft_step_level_verl_multiturn/valid.jsonl \
  --think-file "$THINK_FILE" \
  --out-dir data/processed/sft_step_level_verl_deepseek_think_500 \
  --max-think-chars 800 \
  --require-all \
  --require-action-match

python scripts/data/validate_webshop_sft_dataset.py \
  --train data/processed/sft_step_level_verl_deepseek_think_500/train.jsonl \
  --valid data/processed/sft_step_level_verl_deepseek_think_500/valid.jsonl
