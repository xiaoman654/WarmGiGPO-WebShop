# Server Command Rules

These rules are mandatory when generating commands for server-side training,
evaluation, RL runs, or any long-running experiment.

## 1. Never Give Bare Long-running Commands

Do not provide commands like this for training or evaluation:

```bash
bash scripts/path/to/run.sh
```

Always save stdout and stderr with `tee`:

```bash
set -o pipefail
mkdir -p logs/train logs/eval logs/rl

bash scripts/path/to/run.sh \
  2>&1 | tee logs/train/descriptive_name_$(date +%Y%m%d_%H%M%S).log
```

Use the correct log directory:

- SFT training: `logs/train/`
- Evaluation-only runs: `logs/eval/`
- RL/GiGPO runs: `logs/rl/`
- Smoke tests: use the closest matching directory and include `smoke` in the filename.

## 2. Every Command Block Must Have Three Parts

For non-trivial runs, provide commands in this order:

1. Pre-check commands
2. Run commands with `tee`
3. Post-check commands

Example:

```bash
cd /root/autodl-fs/WarmGiGPO-WebShop
set -o pipefail

ls -lh outputs/sft/qwen25_1p5b_webshop_lora_verl_500/merged_model

mkdir -p logs/eval
bash scripts/eval/run_qwen15b_sft500_eval64.sh \
  2>&1 | tee logs/eval/qwen15b_sft500_eval64_$(date +%Y%m%d_%H%M%S).log

latest=$(ls -t logs/eval/qwen15b_sft500_eval64_*.log | head -1)
grep -E "Initial validation metrics|Final validation metrics|step:0|val/text/test_score|val/success_rate|val/webshop_task_score" \
  "$latest"
```

## 3. Use Descriptive Log Names

Log filenames must include:

- model or setting name
- training/eval mode
- data scale when relevant
- timestamp

Good examples:

```text
logs/train/qwen15b_lora_sft_verl_500_20260519_210000.log
logs/eval/qwen15b_sft2k_eval64_20260519_213000.log
logs/rl/qwen15b_sft500_gigpo_128_64_20260519_220000.log
```

Bad examples:

```text
logs/run.log
logs/output.txt
```

## 4. Check Artifacts Before Re-running Expensive Jobs

Before asking to rerun SFT or merge jobs, first check whether the outputs already
exist.

```bash
ls -lh outputs/sft/qwen25_1p5b_webshop_lora_verl_500/final_adapter
ls -lh outputs/sft/qwen25_1p5b_webshop_lora_verl_500/merged_model
du -sh outputs/sft/qwen25_1p5b_webshop_lora_verl_500
```

If the artifacts exist, avoid retraining unless the previous run is known to be
wrong.

## 5. Extract Metrics Immediately After Each Run

Prefer checking the newest log from the run that just finished, instead of
grepping all historical logs. This prevents old experiments from being mixed
into the result.

After SFT training:

```bash
latest=$(ls -t logs/train/qwen15b_lora_sft_verl_500_*.log | head -1)
grep -E "train_rows|valid_rows|eval_loss|train_runtime|train_loss|saved final adapter" \
  "$latest"
```

After SFT-only eval:

```bash
latest=$(ls -t logs/eval/qwen15b_sft500_eval64_*.log | head -1)
grep -E "Initial validation metrics|step:0|Final validation metrics|val/text/test_score|val/success_rate|val/webshop_task_score" \
  "$latest"
```

After RL/GiGPO:

```bash
latest=$(ls -t logs/rl/qwen15b_sft500_gigpo_128_64_*.log | head -1)
grep -E "Initial validation metrics|Final validation metrics|step:0|step:8|step:16|step:24|step:32|valid_action_ratio|response_length/mean|response_length/clip_ratio|val/text/test_score|val/success_rate|val/webshop_task_score|perf/max_memory" \
  "$latest"
```

When comparing several runs, explicitly list their log patterns:

```bash
grep -E "Final validation metrics|step:32|val/text/test_score|val/success_rate|val/webshop_task_score" \
  logs/rl/qwen15b_sft500_gigpo_128_64_*.log \
  logs/rl/qwen15b_sft2k_gigpo_128_64_*.log
```

## 6. One Expensive GPU Run at a Time

Do not run multiple A800 RL experiments in parallel unless explicitly requested.
The previous 128/64 runs used roughly 65-76 GB reserved GPU memory, so parallel
runs are risky.

Before starting another GPU job, check whether an old run is still active:

```bash
ps aux | grep -E "main_ppo|TaskRunner|WorkerDict|actor_rollout|vllm|sft_lora.py|python" | grep -v grep
nvidia-smi
```

## 7. Preserve Results Before Shutdown

Before stopping a server, run:

```bash
cd /root/autodl-fs/WarmGiGPO-WebShop

git status --short
ls -lh logs/train logs/eval logs/rl 2>/dev/null
ls -lh reports/tables reports/analysis 2>/dev/null
```

If important logs are untracked but should be kept, either commit the summary
reports or copy the exact log files needed for analysis.

## 8. If Logs Were Not Saved

Try to recover Ray output:

```bash
grep -R -A3 -B2 "Final validation metrics" /tmp/ray/session_latest/logs/worker-*.out | tail -80
grep -R "step:32" /tmp/ray/session_latest/logs/worker-*.out | tail -5
```

Recovered Ray output should be treated as provisional unless the experiment name
or model path can be identified.
