# GiGPO Smoke Runs

These scripts run tiny WebShop GiGPO smoke tests on one GPU.

They are not formal experiments. Their purpose is to verify that both the
original model and the SFT warm-start checkpoint can enter the GiGPO training
loop with the same configuration.

Before running the SFT warm-start smoke, merge the LoRA adapter:

```bash
cd /root/autodl-fs/WarmGiGPO-WebShop

MODEL_DIR=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

python scripts/train/merge_lora_adapter.py \
  --base-model "$MODEL_DIR" \
  --adapter outputs/sft/qwen25_1p5b_webshop_lora_verl_full/final_adapter \
  --output-dir outputs/sft/qwen25_1p5b_webshop_lora_verl_full/merged_model
```

Run original Qwen GiGPO smoke:

```bash
mkdir -p logs/rl
bash scripts/rl/run_qwen15b_gigpo_smoke.sh \
  2>&1 | tee logs/rl/qwen15b_gigpo_smoke_$(date +%Y%m%d_%H%M%S).log
```

Run SFT warm-start GiGPO smoke:

```bash
mkdir -p logs/rl
bash scripts/rl/run_qwen15b_sft_verl_gigpo_smoke.sh \
  2>&1 | tee logs/rl/qwen15b_sft_verl_gigpo_smoke_$(date +%Y%m%d_%H%M%S).log
```

Extract key metrics:

```bash
grep -E "Initial validation metrics|Final validation metrics|Avg size of step-level group|valid_action_ratio|response_length/clip_ratio|val/text/test_score|val/success_rate|val/webshop_task_score|timing_s/testing|perf/max_memory" \
  logs/rl/*.log
```

## Small Comparison

Create a shared small split:

```bash
python scripts/rl/prepare_rl_small_data.py \
  --train-size 16 \
  --val-size 16 \
  --out-dir /root/data/verl-agent/text_rl_small
```

Run original GiGPO:

```bash
mkdir -p logs/rl
bash scripts/rl/run_qwen15b_gigpo_small.sh \
  2>&1 | tee logs/rl/qwen15b_gigpo_small_$(date +%Y%m%d_%H%M%S).log
```

Run SFT warm-start GiGPO:

```bash
mkdir -p logs/rl
bash scripts/rl/run_qwen15b_sft_verl_gigpo_small.sh \
  2>&1 | tee logs/rl/qwen15b_sft_verl_gigpo_small_$(date +%Y%m%d_%H%M%S).log
```

## Medium Comparison

Create a shared medium split:

```bash
python scripts/rl/prepare_rl_small_data.py \
  --train-size 64 \
  --val-size 32 \
  --out-dir /root/data/verl-agent/text_rl_medium
```

Run SFT warm-start first:

```bash
mkdir -p logs/rl
bash scripts/rl/run_qwen15b_sft_verl_gigpo_medium.sh \
  2>&1 | tee logs/rl/qwen15b_sft_verl_gigpo_medium_$(date +%Y%m%d_%H%M%S).log
```

Then run original GiGPO:

```bash
mkdir -p logs/rl
bash scripts/rl/run_qwen15b_gigpo_medium.sh \
  2>&1 | tee logs/rl/qwen15b_gigpo_medium_$(date +%Y%m%d_%H%M%S).log
```
