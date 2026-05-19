# SFT Data Size Ablation Template

Date: TODO  
Model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
RL algorithm: GiGPO  
RL train/eval split: 128/64  
Rollout n: 2  
Max episode steps: 5  
Training steps: 32  

## SFT Training Runs

| SFT size | train rows | valid rows | eval_loss | train_loss | runtime_s | adapter |
|---|---:|---:|---:|---:|---:|---|
| 500 | TODO | TODO | TODO | TODO | TODO | `outputs/sft/qwen25_1p5b_webshop_lora_verl_500/final_adapter` |
| 2k | TODO | TODO | TODO | TODO | TODO | `outputs/sft/qwen25_1p5b_webshop_lora_verl_2k/final_adapter` |
| full | 13625 | 500 eval subset | 0.0582324378 | 0.0757816708 | 4795.5802 | `outputs/sft/qwen25_1p5b_webshop_lora_verl_full/final_adapter` |

## SFT-only Eval64

| Method | val/text/test_score | success_rate | webshop_task_score |
|---|---:|---:|---:|
| zero-shot | 0.1286173633 | 0.015625 | 0.0646577381 |
| SFT-500 | TODO | TODO | TODO |
| SFT-2k | TODO | TODO | TODO |
| SFT-full | 0.0961538462 | 0.015625 | 0.0582589286 |

## SFT + GiGPO 128/64

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 0.2893890675 | 0.046875 | 0.0511363636 | 0.974 | 73.487 | 0.026 |
| SFT-500 + GiGPO | TODO | TODO | TODO | TODO | TODO | TODO |
| SFT-2k + GiGPO | TODO | TODO | TODO | TODO | TODO | TODO |
| SFT-full + GiGPO | 3.2638888889 | 0.328125 | 0.5605305285 | 1.000 | 21.919 | 0.000 |

## Interpretation

TODO: Use this table to separate whether more imitation data improves SFT-only behavior, RL warm-start quality, or both. The main expected signal is not necessarily SFT-only score; the project hypothesis is that SFT improves the GiGPO rollout distribution and makes RL updates more useful.
