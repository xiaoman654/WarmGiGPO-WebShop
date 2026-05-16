# SFT Dataset Validation Notes

Date: 2026-05-16

Validation command:
python scripts/data/validate_webshop_sft_dataset.py \
  --train data/processed/sft_step_level/train.jsonl \
  --valid data/processed/sft_step_level/valid.jsonl

Result:
- errors: 0
- warnings: 37

Conclusion:
The dataset is usable for first-round SFT.

Warning explanation:
Most warnings are suspicious action format cases caused by product titles containing square brackets inside click[...] actions, e.g. item titles with [2022 newest] or [3p]. These are valid WebShop-style actions and are not blocking.

Next:
Use this dataset for LoRA SFT smoke training.
