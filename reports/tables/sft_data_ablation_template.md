# SFT Data Size Ablation Result

Date: 2026-05-19
Model: Qwen2.5-1.5B-Instruct
Environment: WebShop
RL algorithm: GiGPO
RL train/eval split: 128/64
Rollout n: 2
Max episode steps: 5
Training steps: 32

## SFT Training Runs

| SFT size | train rows | valid rows | eval_loss | train_loss | runtime_s | artifact |
|---|---:|---:|---:|---:|---:|---|
| 500 | 500 | 500 eval subset | not captured in tee log | not captured in tee log | not captured in tee log | `outputs/sft/qwen25_1p5b_webshop_lora_verl_500/final_adapter` |
| 2k | 2000 | 500 eval subset | not captured in tee log | not captured in tee log | not captured in tee log | `outputs/sft/qwen25_1p5b_webshop_lora_verl_2k/final_adapter` |
| full | 13625 | 500 eval subset | 0.0582324378 | 0.0757816708 | 4795.5802 | `outputs/sft/qwen25_1p5b_webshop_lora_verl_full/final_adapter` |

The SFT-500 and SFT-2k training commands were initially run without `tee`, so
their terminal loss logs were not preserved in `logs/train/`. Their adapter and
merged-model artifacts were verified on the server before eval/RL runs.

## SFT-only Eval64

| Method | val/text/test_score | success_rate | webshop_task_score |
|---|---:|---:|---:|
| zero-shot | 0.1286173633 | 0.015625 | 0.0646577381 |
| SFT-500 | 0.1577287066 | 0.015625 | 0.0390625 |
| SFT-2k | 0.0000000000 | 0.000000 | 0.0229910714 |
| SFT-full | 0.0961538462 | 0.015625 | 0.0582589286 |

## SFT + GiGPO 128/64

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 0.2893890675 | 0.046875 | 0.0511363636 | 0.974 | 73.487 | 0.026 |
| SFT-500 + GiGPO | 0.0955414013 | 0.015625 | 0.0429315476 | 0.974 | 25.579 | 0.000 |
| SFT-2k + GiGPO | 1.0309278351 | 0.109375 | 0.3129360570 | 1.000 | 21.429 | 0.000 |
| SFT-full + GiGPO | 3.2638888889 | 0.328125 | 0.5605305285 | 1.000 | 21.919 | 0.000 |

## Runtime And Memory

| Method | max_memory_allocated_gb | max_memory_reserved_gb | testing_s | step_s |
|---|---:|---:|---:|---:|
| SFT-500 + GiGPO | 65.750 | 74.678 | 345.669 | 366.526 |
| SFT-2k + GiGPO | 66.503 | 76.404 | 319.275 | 339.159 |
| SFT-full + GiGPO | 65.877 | 75.424 | 315.790 | 337.398 |

## Source Logs

- `logs/eval/qwen15b_sft500_eval64_20260519_210255.log`
- `logs/eval/qwen15b_sft2k_eval64_20260519_211017.log`
- `logs/rl/qwen15b_sft500_gigpo_128_64_20260519_211912.log`
- `logs/rl/qwen15b_sft2k_gigpo_128_64_20260519_220207.log`
- `logs/rl/qwen15b_sft_verl_gigpo_medium_128_64_20260518_141425.log`
- `logs/rl/qwen15b_gigpo_medium_128_64_20260518_145752.log`

## Interpretation

The SFT data-size ablation shows a clear threshold effect. SFT-500 is enough to
keep responses short and mostly valid, but it does not improve GiGPO over the
direct GiGPO baseline. SFT-2k provides a much stronger warm-start and reaches
0.3129 WebShop task score and 10.94% success rate. The full SFT set remains the
strongest result, reaching 0.5605 task score and 32.81% success rate.

This suggests that the benefit of SFT warm-start is not just output-format
alignment. A small amount of demonstration data can stabilize action formatting,
but more step-level demonstrations are needed to learn useful WebShop search and
selection behavior that GiGPO can amplify.
