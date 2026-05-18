# GiGPO Smoke Comparison

Date: 2026-05-18
Model: Qwen2.5-1.5B-Instruct
Environment: WebShop
Algorithm: GiGPO
Config:
- train data: text_tiny
- val data: text_tiny
- env.max_steps: 5
- env.rollout.n: 2
- total train steps: 4
- GPU: A800 80GB

## Original Qwen + GiGPO Smoke

Metrics:
- episode/valid_action_ratio: 0.500
- response_length/mean: 108.300
- response_length/clip_ratio: 0.400
- val/text/test_score: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.0
- timing_s/testing: 59.616
- timing_s/step: 71.931
- perf/max_memory_allocated_gb: 53.869
- perf/max_memory_reserved_gb: 66.596

## SFT-v2 + GiGPO Smoke

Metrics:
- episode/valid_action_ratio: 1.000
- response_length/mean: 20.000
- response_length/clip_ratio: 0.000
- val/text/test_score: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.0
- perf/max_memory_reserved_gb: 66.414

## Interpretation

Both original Qwen and SFT-v2 warm-start can run through the GiGPO pipeline. The SFT-v2 warm-start substantially improves action-format behavior: valid action ratio increases from 0.5 to 1.0, and response clipping drops from 0.4 to 0.0. This supports using the SFT-v2 checkpoint for the main SFT+GiGPO comparison.
