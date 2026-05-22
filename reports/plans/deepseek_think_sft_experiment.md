# DeepSeek-Think SFT Follow-up Experiment

## Goal

Test whether generated current-step reasoning improves the SFT warm-start beyond
empty-think action-format alignment.

The data format follows train-inference consistency:

```text
Historical assistant turns:
<action>{previous_action}</action>

Current assistant turn:
<think>{generated current-step reasoning}</think>
<action>{target_action}</action>
```

Historical `<think>` is intentionally dropped. The model should not depend on
previous hidden reasoning unless inference also preserves it.

## First Scale

Start with 500 train samples and 500 validation samples. Do not generate 2k/full
reasoning until this small set is inspected and tested.

## Workflow

### 1. Build bounded multi-turn base data

```bash
bash scripts/data/build_webshop_sft_multiturn_verl.sh
```

### 2. Export DeepSeek requests

```bash
bash scripts/data/export_webshop_deepseek_think_requests_500.sh
```

This writes:

```text
data/processed/deepseek_think_requests/webshop_multiturn_train500_requests.jsonl
data/processed/deepseek_think_requests/webshop_multiturn_valid500_requests.jsonl
data/processed/deepseek_think_requests/webshop_multiturn_500_requests.jsonl
```

Each row contains:

- `sample_id`
- `instruction`
- `history`
- `observation`
- `available_actions`
- `prompt`

By default, `target_action` is intentionally not exported and not included in
the prompt. This avoids answer leakage: DeepSeek should simulate the current
decision rather than rationalize a revealed label. A rationalization mode exists
behind `--include-target-action`, but it is not the recommended setting for this
experiment.

Send `webshop_multiturn_500_requests.jsonl` to the generator. The response JSONL
should preserve `sample_id`, include generated reasoning, and ideally include
DeepSeek's own chosen action for quality checking. The merge script accepts
reasoning from one of these fields:

```text
think
reasoning
generated_think
content
response
text
```

Recommended output row:

```json
{
  "sample_id": "traj_00001_step_002",
  "think": "The current page shows the matching option, so selecting it advances toward purchase.",
  "chosen_action": "click[option]"
}
```

If the generator returns the JSON object as a string under `content`,
`response`, or `text`, the merge script will parse that as well.

To call the DeepSeek API directly:

```bash
export DEEPSEEK_API_KEY=...
# Defaults: deepseek-v4-pro, no hidden API thinking fields, max_tokens=1024, temperature=0.1.
# Override with DEEPSEEK_MODEL / DEEPSEEK_THINKING / DEEPSEEK_REASONING_EFFORT if needed.

# Generate a small quality-check slice first.
MAX_SAMPLES=20 bash scripts/data/generate_webshop_deepseek_think_500.sh

# Resume and finish all requests after inspection.
bash scripts/data/generate_webshop_deepseek_think_500.sh
```

The API script writes to:

```text
data/processed/deepseek_think_requests/webshop_multiturn_500_responses.jsonl
```

It is resumable and skips existing `sample_id`s by default.
Empty responses, truncated JSON, and rows without a parseable `chosen_action`
are treated as invalid and retried. The wrapper omits DeepSeek's hidden
`thinking` field by default because this experiment needs visible reasoning in
the returned JSON; hidden thinking can consume the output budget and cause empty
or truncated final content.

### 3. Inspect before training

Before SFT, inspect 30-50 generated rows:

- concise, preferably under 80 words
- grounded in the current observation
- no invented product attributes
- no mention of "ground truth", "label", or "target_action"
- no `<action>` in the reasoning field
- if `chosen_action` is present, it should usually match the supervised
  `target_action`; the merge step reports match rates in `stats.json`

### 4. Merge generated think into SFT data

Put the generated responses at:

```text
data/processed/deepseek_think_requests/webshop_multiturn_500_responses.jsonl
```

Then run:

```bash
bash scripts/data/build_webshop_sft_deepseek_think_500.sh
```

This writes:

```text
data/processed/sft_step_level_verl_deepseek_think_500/train.jsonl
data/processed/sft_step_level_verl_deepseek_think_500/valid.jsonl
data/processed/sft_step_level_verl_deepseek_think_500/stats.json
```

The merge uses `--require-all`, so only rows with generated think are kept.
It still uses the human-demonstration `target_action` as the final SFT action.
Generated `chosen_action` is recorded only as a data-quality signal unless
`--require-action-match` is enabled manually.

### 5. Train and evaluate

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

## Compare Against

- single-turn empty-think SFT-500
- bounded multi-turn empty-think SFT-500
- DeepSeek-think SFT-500

Main metrics:

- SFT-only eval64 score/success/task score
- SFT+GiGPO 128/64 final metrics
- response length and clipping
- valid action ratio
- KL loss during RL

## Decision Rule

Scale to 2k only if the 500-sample DeepSeek-think version improves either:

- SFT-only behavior without causing long/unstable output, or
- SFT+GiGPO final success/task score over empty-think SFT-500.
