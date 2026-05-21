#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WarmGiGPO-WebShop}

cd "$PROJECT_DIR"

python scripts/data/build_webshop_sft_dataset.py \
  --input data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl \
  --out-dir data/processed/sft_step_level_verl_multiturn \
  --valid-ratio 0.1 \
  --seed 42 \
  --split-by trajectory \
  --target-format verl \
  --conversation-mode multi_turn \
  --multiturn-history-window 2 \
  --history-assistant action_only

python scripts/data/validate_webshop_sft_dataset.py \
  --train data/processed/sft_step_level_verl_multiturn/train.jsonl \
  --valid data/processed/sft_step_level_verl_multiturn/valid.jsonl
