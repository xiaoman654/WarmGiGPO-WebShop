#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WarmGiGPO-WebShop}
REQUEST_FILE=${REQUEST_FILE:-data/processed/deepseek_think_requests/webshop_multiturn_500_requests.jsonl}
OUTPUT_FILE=${OUTPUT_FILE:-data/processed/deepseek_think_requests/webshop_multiturn_500_responses.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-}

cd "$PROJECT_DIR"

if [ ! -f "$REQUEST_FILE" ]; then
  bash scripts/data/export_webshop_deepseek_think_requests_500.sh
fi

args=(
  --input "$REQUEST_FILE"
  --output "$OUTPUT_FILE"
)

if [ -n "$MAX_SAMPLES" ]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi

python scripts/data/generate_deepseek_think_responses.py "${args[@]}"

echo "wrote responses: $OUTPUT_FILE"
