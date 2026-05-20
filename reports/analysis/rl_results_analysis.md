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

## Curve-Level Observations

The RL metric curves generated from the 128/64 logs provide a more detailed view of how the performance gap emerges.

Recommended main-text figures:

- `reports/figures/rl_metrics/val_success_rate.png`
- `reports/figures/rl_metrics/val_webshop_task_score_not_success_rate.png`
- `reports/figures/rl_metrics/episode_valid_action_ratio.png`
- `reports/figures/rl_metrics/response_length_mean.png`
- `reports/figures/rl_metrics/response_length_clip_ratio.png`
- `reports/figures/rl_metrics/step_level_group_size.png`

The validation curves show that SFT + GiGPO improves steadily across validation checkpoints, while direct GiGPO remains much flatter. In the final validation, SFT + GiGPO reaches 0.3281 success rate and 0.5605 WebShop task score, compared with 0.0469 success rate and 0.0511 task score for direct GiGPO.

The behavior curves show a clearer mechanism:

- SFT + GiGPO keeps valid action ratio close to 1.0 for almost the whole run.
- Direct GiGPO starts with much lower valid action ratio and only becomes more stable later.
- SFT + GiGPO keeps response length near 20-30 tokens.
- Direct GiGPO often produces 70-100 token responses.
- SFT + GiGPO almost never clips responses, while direct GiGPO has visible clipping in multiple steps.

These curves support the interpretation that SFT improves the quality of rollout trajectories before and during RL.

## Efficiency Observations

The `timing_s_step.png` curve should be interpreted carefully. Steps with validation are much slower because WebShop evaluation is expensive. This is expected under the current configuration:

- normal training steps are usually around 20-35 seconds,
- validation steps can take several hundred seconds.

Therefore, the main runtime bottleneck is not only model update but also periodic WebShop validation. For larger experiments, reducing validation frequency or validation set size is one of the simplest ways to control cost.

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

## Contribution Decomposition

The eval64 decomposition compares zero-shot, SFT-only, direct GiGPO, and SFT + GiGPO on the same evaluation split:

| Method | val/text/test_score | success_rate | webshop_task_score |
|---|---:|---:|---:|
| Zero-shot | 0.1286 | 0.0156 | 0.0647 |
| SFT-only | 0.0962 | 0.0156 | 0.0583 |
| GiGPO | 0.2894 | 0.0469 | 0.0511 |
| SFT + GiGPO | 3.2639 | 0.3281 | 0.5605 |

This decomposition is important because SFT-only does not directly outperform zero-shot. The main gain appears only when SFT is used as the initialization for GiGPO. Therefore, the current evidence should be interpreted as an RL warm-start effect rather than a standalone supervised-learning improvement.

In other words, SFT does not make the model a strong WebShop agent by itself. It makes the model a better starting policy for GiGPO by improving action format, output length, and rollout parseability.

## SFT Data Size Ablation

We further tested whether the SFT warm-start effect depends on the amount of
step-level demonstration data. The SFT-500, SFT-2k, and SFT-full models use the
same verl-aligned target format and the same downstream GiGPO 128/64 setting.

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 0.2894 | 0.0469 | 0.0511 | 0.974 | 73.487 | 0.026 |
| SFT-500 + GiGPO | 0.0955 | 0.0156 | 0.0429 | 0.974 | 25.579 | 0.000 |
| SFT-2k + GiGPO | 1.0309 | 0.1094 | 0.3129 | 1.000 | 21.429 | 0.000 |
| SFT-full + GiGPO | 3.2639 | 0.3281 | 0.5605 | 1.000 | 21.919 | 0.000 |

The ablation shows a threshold effect. SFT-500 stabilizes output length and
formatting but does not improve final WebShop score over direct GiGPO. SFT-2k
already provides a meaningful policy prior, improving task score to 0.3129 and
success rate to 0.1094. The full SFT set further improves both metrics.

This supports a more precise interpretation of the warm-start effect: a small
amount of imitation data can teach the action protocol, but more demonstrations
are needed to learn WebShop-specific search and selection behavior that GiGPO
can build on.

## GiGPO-Specific Implications

GiGPO relies on useful trajectory variation and anchor state grouping. SFT can affect this mechanism in two competing ways:

- Positive effect: fewer invalid actions and cleaner repeated states make step-level comparisons more meaningful.
- Potential negative effect: stronger imitation prior may reduce action diversity and exploration.

In the current 128/64 setting, the positive effect appears dominant. SFT + GiGPO improves both success rate and task score while maintaining stable action formatting.

The step-level group size curves show that average group size mostly stays around 1.5-2.3, which is reasonable under `env.rollout.n=2`. Direct GiGPO tends to show slightly larger or more variable group sizes, while SFT + GiGPO is somewhat more concentrated. This is consistent with the idea that SFT narrows the action distribution and makes rollouts more regular, while still leaving enough variation for GiGPO's step-level comparison.

## KL Reference Ablation

We tested whether the SFT warm-start effect comes only from actor
initialization, or whether the KL reference policy also matters. In this
ablation, the actor and rollout are still initialized from the full SFT merged
checkpoint, but the reference policy is changed from the SFT checkpoint to the
original Qwen2.5-1.5B-Instruct base model.

| Actor init | KL reference | val/text/test_score | success_rate | webshop_task_score | actor/kl_loss | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SFT-full | SFT-full | 3.2639 | 0.3281 | 0.5605 | 0.020 | 1.000 | 21.919 | 0.000 |
| SFT-full | Base Qwen | 0.0000 | 0.0000 | 0.0000 | 1.394 | 1.000 | 23.838 | 0.000 |

The base-reference run completed successfully, but final validation collapsed
to zero. The key signal is `actor/kl_loss`: it rises from about 0.020 in the
SFT-reference run to 1.394 when the reference is the original base model. This
indicates a strong mismatch between the SFT-initialized WebShop policy and the
base-model reference distribution.

The result strengthens the mechanism story. SFT is not just a better starting
point for the actor. It also provides a better KL anchor for preserving the
WebShop-specific action prior during GiGPO. When the reference is changed back
to the original model, the policy still emits short and valid actions, but the
optimization is constrained toward a pre-SFT distribution and fails to preserve
the task-performance gains.

## System Bottleneck Summary

The timing logs show that WebShop Agentic RL is rollout and validation heavy.
Normal 128/64 training steps take about 21-25 seconds, and actor update accounts
for only about 30-32% of that time. Validation steps are much more expensive:
the final validation steps take about 337-437 seconds, with `timing_s/testing`
accounting for roughly 94% of the total step time.

This means that future engineering work should focus less on optimizing the
actor update and more on environment interaction and rollout throughput:
per-step latency profiling, observation caching, batched or parallel WebShop
environment stepping, and eventually semi-async rollout collection.

See `reports/analysis/system_bottleneck_analysis.md` for the detailed timing
tables.

The next analysis should extract:

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
5. Add a system bottleneck analysis for rollout, validation, and environment interaction time.
