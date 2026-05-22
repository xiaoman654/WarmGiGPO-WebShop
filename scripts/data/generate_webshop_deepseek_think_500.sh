#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WarmGiGPO-WebShop}
REQUEST_FILE=${REQUEST_FILE:-data/processed/deepseek_think_requests/webshop_multiturn_500_requests.jsonl}
OUTPUT_FILE=${OUTPUT_FILE:-data/processed/deepseek_think_requests/webshop_multiturn_500_responses.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-}
DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro}
DEEPSEEK_THINKING=${DEEPSEEK_THINKING:-none}
DEEPSEEK_REASONING_EFFORT=${DEEPSEEK_REASONING_EFFORT:-none}
DEEPSEEK_RESPONSE_FORMAT=${DEEPSEEK_RESPONSE_FORMAT:-json_object}
DEEPSEEK_MAX_TOKENS=${DEEPSEEK_MAX_TOKENS:-512}
DEEPSEEK_TEMPERATURE=${DEEPSEEK_TEMPERATURE:-0.05}
DEEPSEEK_RETRIES=${DEEPSEEK_RETRIES:-5}
FAILURE_FILE=${FAILURE_FILE:-data/processed/deepseek_think_requests/webshop_multiturn_500_failures.jsonl}
INVALID_FILE=${INVALID_FILE:-data/processed/deepseek_think_requests/webshop_multiturn_500_invalid.jsonl}

cd "$PROJECT_DIR"

if [ ! -f "$REQUEST_FILE" ]; then
  bash scripts/data/export_webshop_deepseek_think_requests_500.sh
fi

args=(
  --input "$REQUEST_FILE"
  --output "$OUTPUT_FILE"
  --model "$DEEPSEEK_MODEL"
  --thinking "$DEEPSEEK_THINKING"
  --reasoning-effort "$DEEPSEEK_REASONING_EFFORT"
  --response-format "$DEEPSEEK_RESPONSE_FORMAT"
  --max-tokens "$DEEPSEEK_MAX_TOKENS"
  --temperature "$DEEPSEEK_TEMPERATURE"
  --retries "$DEEPSEEK_RETRIES"
  --failure-output "$FAILURE_FILE"
  --invalid-output "$INVALID_FILE"
)

if [ -n "$MAX_SAMPLES" ]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi

python scripts/data/generate_deepseek_think_responses.py "${args[@]}"

echo "wrote responses: $OUTPUT_FILE"
