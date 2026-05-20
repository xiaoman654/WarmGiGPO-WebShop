# System Bottleneck Analysis

Date: 2026-05-20  
Project: SFT-GiGPO / WarmGiGPO-WebShop  
Setting: WebShop train128/eval64 GiGPO runs  

## Summary

The 128/64 runs show that the main wall-clock bottleneck is not actor update.
Normal training steps take roughly 21-25 seconds, while validation steps take
roughly 337-437 seconds. On validation steps, `timing_s/testing` alone accounts
for about 94% of total step time.

This supports a system-level interpretation: WebShop Agentic RL is dominated by
rollout, validation, and environment interaction, not by the gradient update
itself.

## Normal Training Step Averages

The table below averages non-validation training steps from the local 128/64
logs.

| Run | avg step_s | avg gen_s | avg old_log_prob_s | avg ref_s | avg update_actor_s | update_actor / step |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 24.758 | 9.075 | 2.195 | 6.213 | 7.244 | 29.3% |
| SFT-full + GiGPO | 21.539 | 6.554 | 2.113 | 5.966 | 6.875 | 31.9% |
| SFT-500 + GiGPO | 21.786 | 6.561 | 2.144 | 6.120 | 6.931 | 31.8% |
| SFT-2k + GiGPO | 21.427 | 6.504 | 2.115 | 6.036 | 6.743 | 31.5% |

The actor update is only about 30-32% of a normal training step. Generation,
reference log probability, old log probability, reward processing, advantage
computation, and environment orchestration take the rest.

## Validation Step Cost

The table below uses the final validation step from each run.

| Run | final step_s | testing_s | testing / step | gen_s | ref_s | update_actor_s | max_memory_allocated_gb |
|---|---:|---:|---:|---:|---:|---:|---:|
| GiGPO | 437.004 | 413.634 | 94.7% | 8.376 | 5.941 | 6.916 | 66.071 |
| SFT-full + GiGPO | 337.398 | 315.790 | 93.6% | 6.474 | 5.635 | 7.421 | 65.877 |
| SFT-500 + GiGPO | 366.526 | 345.669 | 94.3% | 6.413 | 5.773 | 6.613 | 65.750 |
| SFT-2k + GiGPO | 339.159 | 319.275 | 94.1% | 6.792 | 5.290 | 5.934 | 66.503 |
| SFT-full + GiGPO, ref=base | 373.156 | 351.660 | 94.2% | 6.799 | 5.599 | 7.098 | 66.578 |

The validation cost is the dominant factor in total runtime. This is expected
because each validation checkpoint runs WebShop interaction over the evaluation
set, and each task requires multi-step text generation plus environment state
updates.

## Implications

1. Reducing `test_freq` or eval set size is the fastest way to reduce wall-clock
   cost during experimentation.
2. Improving actor update speed alone will not substantially improve end-to-end
   throughput.
3. A future engineering project should focus on rollout and environment
   interaction: profiling, caching, batched environment stepping, and possibly
   semi-async rollout collection.
4. SFT warm-start also has a systems benefit: compared with direct GiGPO, SFT
   runs generate shorter responses and lower generation time per normal step.

## Follow-up Engineering Direction

A separate engineering project can be framed as:

```text
WebShop RL Rollout Profiling and Acceleration
```

Recommended MVP:

1. Add per-env-step latency logging for search, click, observation processing,
   reward computation, and action parsing.
2. Add deterministic caching for repeated WebShop search/click observations.
3. Build a standalone parallel rollout collector to measure environment
   throughput independent of model update.
4. Only after profiling and caching, consider a controlled semi-async rollout
   prototype for GiGPO.
