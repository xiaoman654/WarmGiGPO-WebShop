# KL Reference Ablation

Date: 2026-05-20  
Model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
Algorithm: GiGPO  
Train size: 128  
Eval size: 64  
Max episode steps: 5  
Training steps: 32  

This ablation tests whether the SFT warm-start effect depends only on actor
initialization or also on the KL reference policy. Both SFT-reference and
base-reference runs initialize the actor and rollout model from the full SFT
merged checkpoint. They differ only in the reference policy used by the actor
KL loss.

## Main Result

| Actor init | KL reference | val/text/test_score | success_rate | webshop_task_score | actor/kl_loss | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SFT-full | SFT-full | 3.2639 | 0.3281 | 0.5605 | 0.020 | 1.000 | 21.919 | 0.000 |
| SFT-full | Base Qwen | 0.0000 | 0.0000 | 0.0000 | 1.394 | 1.000 | 23.838 | 0.000 |

## Base-Reference Final Metrics

Log:

```text
logs/rl/qwen15b_sft_verl_gigpo_128_64_ref_base_20260520_102628.log
```

Final validation:

- val/text/test_score: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.0

Final step behavior:

- actor/kl_loss: 1.394
- episode/valid_action_ratio: 1.000
- response_length/mean: 23.838
- response_length/max: 54.000
- response_length/min: 14.000
- response_length/clip_ratio: 0.000
- perf/max_memory_allocated_gb: 66.578
- perf/max_memory_reserved_gb: 74.900
- timing_s/testing: 351.660
- timing_s/step: 373.156

## Interpretation

The result shows that the SFT checkpoint is useful not only as an actor
initialization but also as a KL reference anchor. When the reference policy is
changed back to the original base model, the actor KL loss rises from about
0.020 to 1.394 and final validation performance collapses to zero.

This suggests that using the base model as reference creates a mismatched
constraint: it regularizes the SFT-initialized policy toward the pre-SFT
distribution and penalizes the WebShop-specific action prior learned during
SFT. Even though the policy still produces short and valid actions, the RL
optimization no longer preserves the performance gain observed with the SFT
reference.
