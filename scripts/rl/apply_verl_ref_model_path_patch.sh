#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WarmGiGPO-WebShop}
VERL_DIR="$PROJECT_DIR/third_party/verl-agent"
PATCH_FILE="$PROJECT_DIR/patches/verl-agent-ref-model-path.patch"

cd "$PROJECT_DIR"

if grep -q "model_path: null" "$VERL_DIR/verl/trainer/config/ppo_trainer.yaml" \
  && grep -q "ref_model_path = self.config.ref.get(\"model_path\"" "$VERL_DIR/verl/workers/fsdp_workers.py"; then
  echo "verl-agent ref model path patch already applied."
  exit 0
fi

echo "Applying verl-agent ref model path patch..."
git -C "$VERL_DIR" apply "$PATCH_FILE"
echo "Applied verl-agent ref model path patch."
