# SFT-v2 + GiGPO Smoke Result

Date: 2026-05-17
Model: Qwen2.5-1.5B-Instruct
Init policy: verl-aligned full SFT merged model
Algorithm: GiGPO
Environment: WebShop
Smoke config:
- train data: text_tiny
- val data: text_tiny
- env.max_steps: 5
- env.rollout.n: 2
- total train steps: 4
- GPU: A800 80GB

Result: SUCCESS

Metrics:
- Training Progress: 4/4 completed
- episode/valid_action_ratio: 1.000
- response_length/mean: 20.000
- response_length/clip_ratio: 0.000
- val/text/test_score: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.0
- timing_s/testing: 38.105
- timing_s/step: 48.495
- perf/max_memory_allocated_gb: 53.869
- perf/max_memory_reserved_gb: 66.414

Interpretation:
The SFT-v2 checkpoint can be used as a GiGPO warm-start. The smoke run completes the full rollout, logprob, reference, advantage, and actor update pipeline. Action validity remains perfect, confirming that verl-aligned SFT is compatible with verl-agent WebShop parsing. Final success is 0 in this tiny smoke setting and should not be interpreted as formal performance.
