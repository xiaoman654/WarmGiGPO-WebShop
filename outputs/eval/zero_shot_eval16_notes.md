# Zero-shot Eval16 Notes

Date: 2026-05-16
Model: Qwen2.5-1.5B-Instruct
Environment: WebShop
Framework: verl-agent
Mode: zero-shot baseline via val_before_train
Eval samples: 16
max_steps: 5
temperature: 0.4

Pure zero-shot initial validation:
- val/text/test_score: 0.0
- val/text/tool_call_count/mean: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.020833333333333332

After one dummy training step:
- val/text/test_score: 0.0
- val/text/tool_call_count/mean: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.08660714285714285

Training/diagnostic metrics:
- episode/valid_action_ratio: 0.0
- response_length/clip_ratio: 0.8
- prompt_length/mean: 288.0
- response_length/mean: 124.0
- timing_s/testing: 197.898
- perf/max_memory_allocated_gb: 53.868
- perf/max_memory_reserved_gb: 59.729

Observation:
- Zero-shot success rate is 0.0 on eval16.
- Pure zero-shot WebShop task score is very low.
- The model often reaches the response length cap.
- The training rollout valid_action_ratio is 0.0, suggesting action formatting and next-action imitation are important.
- This supports moving to WebShop human-demo SFT data construction.
