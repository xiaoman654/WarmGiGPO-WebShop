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

## SFT Data Size Ablation

Use these scripts to test whether SFT warm-start quality depends on the number
of WebShop demonstration steps. They keep the same model, LoRA config, seed,
validation subset, and prompt/target format as the full verl-aligned SFT run.

```bash
cd /root/autodl-fs/WarmGiGPO-WebShop
mkdir -p logs/train

bash scripts/train/run_qwen15b_lora_sft_verl_500.sh \
  2>&1 | tee logs/train/qwen15b_lora_sft_verl_500_$(date +%Y%m%d_%H%M%S).log

bash scripts/train/run_qwen15b_lora_sft_verl_2k.sh \
  2>&1 | tee logs/train/qwen15b_lora_sft_verl_2k_$(date +%Y%m%d_%H%M%S).log
```

Merge the ablation adapters before SFT-only eval or RL warm-start:

```bash
bash scripts/train/merge_sft_ablation_adapters.sh \
  2>&1 | tee logs/train/merge_sft_ablation_adapters_$(date +%Y%m%d_%H%M%S).log
```

Expected outputs:

```text
outputs/sft/qwen25_1p5b_webshop_lora_verl_500/
outputs/sft/qwen25_1p5b_webshop_lora_verl_2k/
```

## Multi-turn verl-aligned SFT

This is the first follow-up experiment for testing whether a limited
trajectory-prefix chat history gives a better warm-start than the current
isolated single-step SFT. It keeps the same empty-think target format for the
current turn:

```text
<think></think>
<action>{target_action}</action>
```

Historical assistant turns are stored as action-only by default:

```text
<action>{previous_action}</action>
```

Each row keeps at most the most recent 2 previous turns before the current
trajectory step. This avoids putting historical thinking into the model input
and prevents trajectory-prefix prompts from exploding in length. The SFT trainer
now masks non-assistant tokens and supervises assistant turns for both two-turn
and multi-turn rows.

Build and validate the multi-turn data:

```bash
bash scripts/data/build_webshop_sft_multiturn_verl.sh
```

Run the 500-sample multi-turn SFT:

```bash
mkdir -p logs/train

bash scripts/train/run_qwen15b_lora_sft_verl_multiturn_500.sh \
  2>&1 | tee logs/train/qwen15b_lora_sft_verl_multiturn_500_$(date +%Y%m%d_%H%M%S).log
```

Merge the adapter:

```bash
bash scripts/train/merge_sft_multiturn_adapters.sh \
  2>&1 | tee logs/train/merge_sft_multiturn_adapters_$(date +%Y%m%d_%H%M%S).log
```

Then run SFT-only eval64 and SFT+GiGPO 128/64:

```bash
mkdir -p logs/eval logs/rl

bash scripts/eval/run_qwen15b_sft_multiturn500_eval64.sh \
  2>&1 | tee logs/eval/qwen15b_sft_multiturn500_eval64_$(date +%Y%m%d_%H%M%S).log

bash scripts/rl/run_qwen15b_sft_multiturn500_gigpo_128_64.sh \
  2>&1 | tee logs/rl/qwen15b_sft_multiturn500_gigpo_128_64_$(date +%Y%m%d_%H%M%S).log
```

Compare against the existing single-turn SFT-500 results before scaling to 2k
or adding generated reasoning inside `<think>`.

## DeepSeek-think SFT-500

After generating and merging current-step reasoning data as described in
`reports/plans/deepseek_think_sft_experiment.md`, train the 500-sample
DeepSeek-think SFT variant:

```bash
mkdir -p logs/train logs/eval logs/rl

bash scripts/train/run_qwen15b_lora_sft_verl_deepseek_think_500.sh \
  2>&1 | tee logs/train/qwen15b_lora_sft_verl_deepseek_think_500_$(date +%Y%m%d_%H%M%S).log

bash scripts/train/merge_sft_deepseek_think_adapter.sh \
  2>&1 | tee logs/train/merge_sft_deepseek_think_adapter_$(date +%Y%m%d_%H%M%S).log

bash scripts/eval/run_qwen15b_sft_deepseekthink500_eval64.sh \
  2>&1 | tee logs/eval/qwen15b_sft_deepseekthink500_eval64_$(date +%Y%m%d_%H%M%S).log

bash scripts/rl/run_qwen15b_sft_deepseekthink500_gigpo_128_64.sh \
  2>&1 | tee logs/rl/qwen15b_sft_deepseekthink500_gigpo_128_64_$(date +%Y%m%d_%H%M%S).log
```
