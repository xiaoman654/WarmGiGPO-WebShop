# Contribution Decomposition Eval64

Date: TBD  
Model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
Eval split: `/root/data/verl-agent/text_128_64/test.parquet`  
Eval size: 64  
Max episode steps: 5  

## Goal

This table decomposes the contribution of imitation warm-start and GiGPO RL:

```text
zero-shot -> SFT-only -> GiGPO -> SFT + GiGPO
```

## Results

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT-only | TBD | TBD | TBD | TBD | TBD | TBD |
| GiGPO | 0.2894 | 0.0469 | 0.0511 | 0.974 | 73.487 | 0.026 |
| SFT + GiGPO | 3.2639 | 0.3281 | 0.5605 | 1.000 | 21.919 | 0.000 |

## Interpretation Template

- Zero-shot measures the base instruction model's initial WebShop ability.
- SFT-only measures how much human-demo imitation improves action formatting and basic decision behavior before RL.
- GiGPO measures direct RL from the base model.
- SFT + GiGPO measures whether imitation warm-start improves the RL trajectory distribution and final performance.

After filling the table, report:

- SFT gain over zero-shot.
- GiGPO gain over zero-shot.
- SFT + GiGPO gain over both SFT-only and GiGPO.
- Whether SFT mainly improves action validity, output length, or final task success.

