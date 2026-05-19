# WarmGiGPO-WebShop

This repository contains a small-model Agentic RL post-training study on WebShop:

**SFT-GiGPO: Studying Imitation Warm-Start for Critic-Free LLM Agent Reinforcement Learning in WebShop**

The project studies whether WebShop human-demo SFT warm-start improves GiGPO training for `Qwen2.5-1.5B-Instruct`.

## Core Comparison

```text
Qwen2.5-1.5B-Instruct -> GiGPO
Qwen2.5-1.5B-Instruct -> WebShop human-demo SFT -> GiGPO
```

The goal is not only to compare final WebShop scores, but also to analyze rollout behavior: valid action ratio, response length, response clipping, and GiGPO step-level grouping.

## Current Main Result

Main setting:

```text
Environment: WebShop
Model: Qwen2.5-1.5B-Instruct
Train size: 128
Eval size: 64
Algorithm: GiGPO
Rollout n: 2
Max episode steps: 5
GPU: 1 x A800 80GB
```

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 0.2894 | 0.0469 | 0.0511 | 0.974 | 73.487 | 0.026 |
| SFT + GiGPO | 3.2639 | 0.3281 | 0.5605 | 1.000 | 21.919 | 0.000 |

SFT + GiGPO improves success rate by about 7.0x and WebShop task score by about 11.0x in the current 128/64 setting.

## Repository Structure

```text
scripts/
  data/       WebShop human-demo inspection and SFT dataset construction
  train/      LoRA SFT and adapter merge scripts
  eval/       WebShop eval scripts
  rl/         GiGPO and SFT+GiGPO run scripts
  analysis/   Log parsing and report artifact generation

reports/
  PROJECT_REPORT.md              Main stage report
  analysis/rl_results_analysis.md
  tables/                         Numeric experiment summaries
  figures/rl_metrics/             RL metric curves

logs/
  rl/                              Main RL logs used for analysis
```

Large upstream repositories, raw data, processed data, checkpoints, and model outputs are intentionally ignored by git.

## Report

Start with:

```text
reports/PROJECT_REPORT.md
```

Detailed RL analysis:

```text
reports/analysis/rl_results_analysis.md
```

Main result table:

```text
reports/tables/rl_128_64_main_comparison.md
```

## Key Interpretation

The current evidence suggests that WebShop human-demo SFT provides a useful imitation prior for GiGPO. It improves the quality of rollout trajectories by making actions shorter, more valid, and less likely to be clipped. This cleaner rollout distribution appears to make GiGPO training more effective.

Future extensions include SFT data-size ablations, KL reference ablations, and trajectory-level failure analysis.

