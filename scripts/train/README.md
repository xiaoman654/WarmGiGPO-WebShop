# SFT Training

Run a small LoRA SFT smoke test after generating and validating the SFT data:

```bash
cd /root/autodl-fs/WarmGiGPO-WebShop
git pull --rebase origin main

python scripts/data/validate_webshop_sft_dataset.py \
  --train data/processed/sft_step_level/train.jsonl \
  --valid data/processed/sft_step_level/valid.jsonl

bash scripts/train/run_qwen15b_lora_sft_smoke.sh \
  2>&1 | tee logs/train/qwen15b_lora_sft_smoke_$(date +%Y%m%d_%H%M%S).log
```

The smoke script trains on at most 500 samples for 50 optimizer steps and writes:

```text
outputs/sft/qwen25_1p5b_webshop_lora_smoke/
```

This is only a pipeline check. Once loss, evaluation, and checkpoint saving look sane, create a full SFT script using the same `sft_lora.py` entrypoint without `--max-train-samples` and with a larger `--max-seq-length` if needed.

Run full one-epoch LoRA SFT after regenerating the data with trajectory-level split:

```bash
python scripts/data/build_webshop_sft_dataset.py \
  --input data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl \
  --out-dir data/processed/sft_step_level \
  --valid-ratio 0.1 \
  --seed 42 \
  --split-by trajectory

python scripts/data/validate_webshop_sft_dataset.py \
  --train data/processed/sft_step_level/train.jsonl \
  --valid data/processed/sft_step_level/valid.jsonl

bash scripts/train/run_qwen15b_lora_sft_full.sh \
  2>&1 | tee logs/train/qwen15b_lora_sft_full_$(date +%Y%m%d_%H%M%S).log
```

Merge a trained LoRA adapter before verl-agent evaluation:

```bash
MODEL_DIR=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

python scripts/train/merge_lora_adapter.py \
  --base-model "$MODEL_DIR" \
  --adapter outputs/sft/qwen25_1p5b_webshop_lora_full/final_adapter \
  --output-dir outputs/sft/qwen25_1p5b_webshop_lora_full/merged_model
```
