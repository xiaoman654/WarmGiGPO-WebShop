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
  --seed 42
```

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
