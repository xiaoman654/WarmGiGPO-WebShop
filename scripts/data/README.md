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
data:

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
