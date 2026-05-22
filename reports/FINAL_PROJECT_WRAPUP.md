# WarmGiGPO-WebShop Final Wrap-up

Date: 2026-05-22  
Status: Experimental phase complete  
Main model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
Core method: WebShop human-demo SFT warm-start + GiGPO

## Final Verdict

This project is ready to close. The strongest result is not that SFT alone
solves WebShop, but that WebShop action imitation provides an effective
warm-start and KL reference anchor for GiGPO.

The main completed chain is:

```text
WebShop human demonstrations
-> step-level SFT data
-> Qwen2.5-1.5B LoRA SFT
-> merged SFT model
-> SFT-only eval
-> GiGPO / SFT+GiGPO comparison
-> SFT data-size ablation
-> KL reference ablation
-> rollout/validation profiling
-> DeepSeek rationale distillation negative result
```

## Core Results

### Main RL Comparison

| Method | val/text/test_score | success_rate | webshop_task_score | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|
| Direct GiGPO | 0.2894 | 0.0469 | 0.0511 | 73.487 | 0.026 |
| SFT-full + GiGPO | 3.2639 | 0.3281 | 0.5605 | 21.919 | 0.000 |

SFT-full + GiGPO improves success rate by about 7x over direct GiGPO and
improves WebShop task score by about 11x.

### SFT Data-Size Ablation

| Method | success_rate | webshop_task_score | Interpretation |
|---|---:|---:|---|
| SFT-500 + GiGPO | 0.0156 | 0.0429 | Mostly learns format; not enough strategy prior. |
| SFT-2k + GiGPO | 0.1094 | 0.3129 | Useful WebShop prior starts to emerge. |
| SFT-full + GiGPO | 0.3281 | 0.5605 | Strongest warm-start. |

### KL Reference Ablation

| Actor init | KL reference | success_rate | webshop_task_score | actor/kl_loss |
|---|---|---:|---:|---:|
| SFT-full | SFT-full | 0.3281 | 0.5605 | 0.020 |
| SFT-full | Base Qwen | 0.0000 | 0.0000 | 1.394 |

The SFT model is not only a better actor initialization; it is also the
compatible reference policy. A base-model reference regularizes the SFT actor
toward the wrong pre-SFT distribution.

### DeepSeek Target-Think Negative Result

| Method | Train rows used | success_rate | webshop_task_score |
|---|---:|---:|---:|
| DeepSeek target-think SFT-only | 259 | 0.0000 | 0.0506 |
| DeepSeek target-think SFT + GiGPO | 259 | 0.0156 | 0.0537 |

Teacher-generated rationales did not improve this setting. The current evidence
supports keeping WebShop SFT action-centered:

```text
<think></think>
<action>{target_action}</action>
```

instead of adding small-scale generated reasoning.

## Final Technical Interpretation

The project supports four claims:

1. SFT-only is not a strong WebShop policy, but it changes the rollout
   distribution in a way that makes GiGPO much more effective.
2. The value of SFT comes from parser-aligned action format, shorter responses,
   fewer clipped generations, and better initial search/click behavior.
3. KL reference choice is critical. The reference policy should match the
   SFT-initialized actor distribution.
4. For Qwen2.5-1.5B in WebShop, action-only / empty-think SFT is more robust
   than small-scale teacher rationale distillation.

## Resume Version

**WebShop Agentic RL Post-training for Qwen2.5-1.5B**

- Built an end-to-end WebShop Agent post-training pipeline with Qwen2.5-1.5B,
  including human-demo step-level SFT data construction, LoRA SFT, adapter
  merging, verl-agent evaluation, and GiGPO reinforcement learning.
- Designed parser-aligned SFT targets in the form
  `<think></think><action>...</action>` to stabilize WebShop action generation
  without relying on unverifiable chain-of-thought labels.
- Improved WebShop eval64 success rate from 4.69% with direct GiGPO to 32.81%
  with SFT warm-start + GiGPO, while reducing mean response length from 73.5 to
  21.9 tokens and eliminating response clipping.
- Conducted SFT data-size ablations showing a clear threshold effect: 500
  examples mainly learned output format, 2k examples began to provide useful
  task prior, and full SFT achieved the strongest RL performance.
- Implemented a KL reference ablation and found that using the original base
  model as the reference collapsed performance to 0%, demonstrating that the
  SFT checkpoint is also the appropriate KL anchor for RL.
- Profiled RL timing and found validation/rollout dominated wall-clock cost
  rather than actor updates, motivating evaluation scheduling and rollout
  optimization as future engineering directions.
- Tested DeepSeek-v4-pro target-conditioned rationale distillation as a follow-up
  SFT variant and showed it did not improve performance under the current
  small-model, filtered-data setting.

## Interview Talking Points

### What was the key finding?

SFT did not directly solve WebShop. Its value was as a warm-start for GiGPO:
it made rollouts shorter, cleaner, parser-aligned, and therefore more useful for
critic-free RL.

### Why use empty `<think>`?

The WebShop demonstrations contain actions, not verified reasoning traces.
Training on fabricated reasoning would introduce noisy process supervision.
Empty `<think>` preserves the parser-required format while focusing the loss on
the reliable action label.

### Why did base-reference KL fail?

The actor had already been shifted by SFT into a WebShop-specific action
distribution. A base Qwen reference pulled it back toward pre-SFT behavior,
causing large KL loss and destroying the task performance.

### Why did DeepSeek rationale distillation fail?

The generated rationale data suffered from long multi-turn contexts, so only
259 train rows survived filtering. Also, for a 1.5B model, extra reasoning text
likely diluted the action imitation signal. In this setup, stable action
generation mattered more than natural-language rationale generation.

## Recommended Stop Point

Do not add more experiments to this project unless a paper-style expansion is
required. The current story is complete:

```text
SFT action prior -> better GiGPO rollouts -> higher WebShop success
matched SFT reference -> stable KL regularization
teacher rationale distillation -> no gain in this small-model setting
```

Potential future work should be separated into new projects:

- larger-scale rationale distillation with quality filtering
- think-loss masking ablation
- WebShop rollout/validation acceleration
- larger train/eval splits and multi-seed verification
