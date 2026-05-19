# RL Results Analysis

Date: 2026-05-19  
Project: SFT-GiGPO / WarmGiGPO-WebShop  
Model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
Algorithm: GiGPO  

## Executive Summary

The current experiments support the core hypothesis: WebShop human-demo SFT warm-start substantially improves GiGPO training on WebShop.

The strongest controlled comparison so far is the 128/64 setting:

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 0.2894 | 0.0469 | 0.0511 | 0.974 | 73.487 | 0.026 |
| SFT + GiGPO | 3.2639 | 0.3281 | 0.5605 | 1.000 | 21.919 | 0.000 |

Relative to direct GiGPO, SFT + GiGPO improves success rate by about 7.0x and WebShop task score by about 11.0x.

## Main Finding

SFT improves not only final task performance but also the quality of the rollout data used by GiGPO. The most visible behavioral difference is output control:

- Direct GiGPO still produces long responses, with mean response length 73.487 and occasional clipping.
- SFT + GiGPO produces short, parseable actions, with mean response length 21.919 and zero clipping.
- Both methods can eventually produce valid actions, but SFT + GiGPO reaches and maintains cleaner action formatting more reliably.

This suggests that the imitation prior learned from WebShop demonstrations reduces exploration waste from malformed, overlong, or hard-to-parse actions.

## Evidence From Small Runs

The small comparison already showed the same pattern:

| Method | Initial task score | Final task score | Final success rate | Response length mean | Clip ratio |
|---|---:|---:|---:|---:|---:|
| GiGPO | 0.0208 | 0.0208 | 0.000 | 84.722 | 0.000 |
| SFT + GiGPO | 0.1229 | 0.2801 | 0.125 | 25.684 | 0.000 |

During small-run training, direct GiGPO had unstable valid action ratios, including 0.200, 0.300, 0.450, and 0.600, while SFT + GiGPO stayed near 1.000. This pattern makes the 128/64 result less likely to be a one-off artifact.

## Interpretation

The results are consistent with the following mechanism:

1. Human-demo SFT teaches the model the WebShop interaction protocol.
2. The model enters GiGPO with a stronger action prior.
3. Rollouts become shorter, more valid, and less clipped.
4. GiGPO receives cleaner trajectories and more useful reward/advantage signals.
5. Final task performance improves substantially.

The SFT stage does not solve WebShop by itself, but it changes the starting distribution enough for GiGPO to learn more effectively.

## GiGPO-Specific Implications

GiGPO relies on useful trajectory variation and anchor state grouping. SFT can affect this mechanism in two competing ways:

- Positive effect: fewer invalid actions and cleaner repeated states make step-level comparisons more meaningful.
- Potential negative effect: stronger imitation prior may reduce action diversity and exploration.

In the current 128/64 setting, the positive effect appears dominant. SFT + GiGPO improves both success rate and task score while maintaining stable action formatting.

The next analysis should extract:

- average step-level group size over training,
- valid action ratio curve,
- response length curve,
- reward / success / task score curve,
- examples of successful and failed trajectories.

## Limitations

The current result is strong but not yet a full-scale benchmark:

- Train size is 128 and eval size is 64, smaller than the full WebShop split.
- Max episode steps is 5, shorter than a more realistic 10-15 step setting.
- Only one seed has been tested.
- The current report mainly uses aggregate metrics; trajectory-level failure analysis is still pending.

These limitations do not invalidate the result, but they should be stated clearly in the final report.

## Recommended Next Steps

1. Parse RL logs into CSV files and plot training curves.
2. Add trajectory examples for direct GiGPO and SFT + GiGPO.
3. Run a second seed for the 128/64 comparison if GPU budget allows.
4. Optionally increase max episode steps to 8 for a more realistic WebShop setting.
5. Add a small ablation on KL reference if time allows:
   - SFT + GiGPO with reference = SFT checkpoint.
   - SFT + GiGPO with reference = original Qwen checkpoint.

