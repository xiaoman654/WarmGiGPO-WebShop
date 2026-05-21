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
but stores trajectory-prefix messages as alternating user/assistant turns. This
lets the SFT trainer supervise assistant turns in a chat history instead of only
a single isolated step. In this first version the `<think>` span is still empty;
DeepSeek-generated reasoning can be added later as a separate data-generation
step.

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
