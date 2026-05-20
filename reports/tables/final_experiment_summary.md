# Final Experiment Summary

Date: 2026-05-20  
Model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
Main RL setting: train128/eval64, max_steps=5, env.rollout.n=2, 32 training steps  

This table collects the main completed experiments into one report-ready view.

| Method | Actor init | KL reference | SFT data | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio | Key interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zero-shot | Base Qwen | N/A | 0 | 0.1286 | 0.0156 | 0.0647 | N/A | N/A | N/A | Base model has weak WebShop ability. |
| SFT-only | SFT-full | N/A | full | 0.0962 | 0.0156 | 0.0583 | N/A | N/A | N/A | SFT alone does not solve WebShop. |
| GiGPO | Base Qwen | Base Qwen | 0 | 0.2894 | 0.0469 | 0.0511 | 0.974 | 73.487 | 0.026 | Direct RL learns weakly and remains verbose. |
| SFT-500 + GiGPO | SFT-500 | SFT-500 | 500 | 0.0955 | 0.0156 | 0.0429 | 0.974 | 25.579 | 0.000 | Small SFT mostly teaches formatting, not strategy. |
| SFT-2k + GiGPO | SFT-2k | SFT-2k | 2000 | 1.0309 | 0.1094 | 0.3129 | 1.000 | 21.429 | 0.000 | More SFT data gives a useful WebShop prior. |
| SFT-full + GiGPO | SFT-full | SFT-full | full | 3.2639 | 0.3281 | 0.5605 | 1.000 | 21.919 | 0.000 | Strongest setting; SFT improves rollout quality and RL learning. |
| SFT-full + GiGPO, ref=base | SFT-full | Base Qwen | full | 0.0000 | 0.0000 | 0.0000 | 1.000 | 23.838 | 0.000 | Mismatched base reference destroys the SFT warm-start gain. |

## Main Takeaways

1. The main gain is not from SFT-only performance. SFT-only is similar to or
   slightly worse than zero-shot on eval64.
2. The gain appears when SFT is used as a warm-start for GiGPO. Full SFT +
   GiGPO improves success rate from 0.0469 to 0.3281 over direct GiGPO.
3. SFT data size matters. SFT-500 is not enough; SFT-2k already gives a clear
   improvement, and full SFT is strongest.
4. The KL reference matters. Using the original base model as reference with an
   SFT-initialized actor collapses final validation to zero, suggesting that
   the SFT checkpoint is also the correct policy anchor.
