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

