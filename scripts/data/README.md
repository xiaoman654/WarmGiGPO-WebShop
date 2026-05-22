# WebShop SFT Data Scripts

The raw WebShop human demonstration file is not stored in Git. On the server it was found at:

```text
data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl
```

It contains 1,571 trajectories with fields such as `states`, `actions`, and `actions_translate`.

Inspect the raw file:

```bash
python scripts/data/inspect_webshop_demos.py \
  --input data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl
```

Build step-level SFT data:

```bash
python scripts/data/build_webshop_sft_dataset.py \
  --input data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl \
  --out-dir data/processed/sft_step_level \
  --valid-ratio 0.1 \
  --seed 42 \
  --split-by trajectory
```

For SFT that is aligned with the default `verl-agent` WebShop prompt and
projection parser, use `--target-format verl`:

```bash
python scripts/data/build_webshop_sft_dataset.py \
  --input data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl \
  --out-dir data/processed/sft_step_level_verl \
  --valid-ratio 0.1 \
  --seed 42 \
  --split-by trajectory \
  --target-format verl
```

This keeps `target_action` as the raw WebShop action while setting the
assistant message to:

```text
<think></think>
<action>{target_action}</action>
```

Build the multi-turn verl-aligned variant:

```bash
bash scripts/data/build_webshop_sft_multiturn_verl.sh
```

This writes:

```text
data/processed/sft_step_level_verl_multiturn/train.jsonl
data/processed/sft_step_level_verl_multiturn/valid.jsonl
data/processed/sft_step_level_verl_multiturn/stats.json
```

The multi-turn variant still uses the same verl-style user prompt at each step,
but stores recent trajectory-prefix messages as alternating user/assistant
turns. To avoid train/inference mismatch and context blow-up, the default keeps
only the most recent 2 previous turns and stores historical assistant turns as
action-only:

```text
<action>{previous_action}</action>
```

The current assistant target remains:

```text
<think></think>
<action>{target_action}</action>
```

This is intentionally conservative. DeepSeek-generated reasoning can be added
later for the current turn, while historical turns should still drop `<think>`
unless inference also keeps historical thinking in context.

Export 500 train + 500 validation requests for generated current-step
reasoning:

```bash
bash scripts/data/export_webshop_deepseek_think_requests_500.sh
```

After filling:

```text
data/processed/deepseek_think_requests/webshop_multiturn_500_responses.jsonl
```

with generated reasoning keyed by `sample_id`, build the DeepSeek-think SFT
data.

If you want to generate the responses through the DeepSeek API, set an API key
and run:

```bash
export DEEPSEEK_API_KEY=...
# Defaults: deepseek-v4-pro, JSON mode, user_id=webshop-sft-think,
# no hidden API thinking fields, max_tokens=512, temperature=0.05.
# Override with DEEPSEEK_MODEL / DEEPSEEK_USER_ID / DEEPSEEK_THINKING / DEEPSEEK_REASONING_EFFORT if needed.

# Optional smoke test first.
MAX_SAMPLES=20 bash scripts/data/generate_webshop_deepseek_think_500.sh

# Resume and finish the full request file.
DEEPSEEK_WORKERS=4 DEEPSEEK_RETRIES=2 bash scripts/data/generate_webshop_deepseek_think_500.sh
```

The generator is resumable: existing `sample_id`s in the response JSONL are
skipped unless `--overwrite` is passed to the underlying Python script. Empty
or truncated responses are treated as invalid and will be regenerated on the
next run. Samples that still fail after all retries are recorded in
`data/processed/deepseek_think_requests/webshop_multiturn_500_failures.jsonl`,
and the batch continues. Invalid intermediate responses are recorded in
`data/processed/deepseek_think_requests/webshop_multiturn_500_invalid.jsonl`
with raw content and finish reason for debugging. For this visible-reasoning
data generation task, the wrapper omits DeepSeek's hidden `thinking` field by
default; enabling hidden thinking can consume output budget and make the final
JSON more likely to be empty or truncated.
Use `DEEPSEEK_WORKERS` for parallel API calls. DeepSeek currently documents a
high account-level concurrency limit for `deepseek-v4-pro`, but start with 8-16
workers to avoid local network, logging, or cost surprises, then increase if the
API account behaves well.

Then build the SFT data:

```bash
bash scripts/data/build_webshop_sft_deepseek_think_500.sh
```

See `reports/plans/deepseek_think_sft_experiment.md` for the full workflow and
quality checks.

The request exporter hides `target_action` by default to avoid answer leakage.
It asks the external generator to return current-step reasoning plus an optional
`chosen_action`. The merge step always keeps the human-demonstration
`target_action` as the supervised `<action>` target, while recording whether the
generated `chosen_action` matches it as a quality diagnostic.

Outputs:

```text
data/processed/sft_step_level/train.jsonl
data/processed/sft_step_level/valid.jsonl
data/processed/sft_step_level/stats.json
```

Validate the generated dataset before training:

```bash
python scripts/data/validate_webshop_sft_dataset.py \
  --train data/processed/sft_step_level/train.jsonl \
  --valid data/processed/sft_step_level/valid.jsonl
```

Each JSONL row contains both explicit fields and chat-style `messages`:

```text
instruction + history + observation -> target_action
```

`data/raw/` and `data/processed/` are intentionally ignored by Git.
