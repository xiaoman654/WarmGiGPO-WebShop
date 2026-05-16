# SFT-full Eval16 Result

Date: 2026-05-16
Model: Qwen2.5-1.5B-Instruct
SFT: Full LoRA SFT, trajectory-level split
Eval: WebShop eval16
Max steps: 5

Metrics:
- val/text/test_score: 0.0
- val/text/tool_call_count/mean: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.0
- episode/valid_action_ratio: 0.600
- response_length/mean: 110.800
- response_length/clip_ratio: 0.400
- prompt_length/mean: 538.400
- timing_s/testing: 219.785
- perf/max_memory_allocated_gb: 53.868
- perf/max_memory_reserved_gb: 60.697

Interpretation:
Full SFT substantially improves valid action behavior compared with zero-shot, but eval16 success remains 0. This suggests SFT mainly teaches action format and local imitation, while final WebShop success still requires RL or better search/item selection.
