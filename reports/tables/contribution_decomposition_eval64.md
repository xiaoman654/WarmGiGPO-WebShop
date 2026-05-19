# Contribution Decomposition Eval64

Date: 2026-05-19  
Model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
Eval split: `/root/data/verl-agent/text_128_64/test.parquet`  
Eval size: 64  
Max episode steps: 5  

## Results

| Method | val/text/test_score | success_rate | webshop_task_score |
|---|---:|---:|---:|
| Zero-shot | 0.1286173633 | 0.015625 | 0.0646577381 |
| SFT-only | 0.0961538462 | 0.015625 | 0.0582589286 |
| GiGPO | 0.2893890675 | 0.046875 | 0.0511363636 |
| SFT + GiGPO | 3.2638888889 | 0.328125 | 0.5605305285 |

## Interpretation

SFT-only does not improve WebShop eval64 performance over zero-shot in this setting. Its task score is slightly lower than zero-shot, and success rate is the same.

However, SFT + GiGPO is far stronger than both direct GiGPO and SFT-only. This suggests that the main value of WebShop human-demo SFT is not direct task success, but better RL initialization. SFT teaches the model the WebShop action format and short-action prior, which improves rollout quality and makes GiGPO training more effective.

This supports the project's main hypothesis: imitation warm-start can improve critic-free Agentic RL by improving the quality of exploration and action generation, even when the SFT policy alone is not a strong final policy.
