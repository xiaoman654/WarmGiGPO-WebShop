# SFT-smoke Eval16 Result

Date: 2026-05-16
Model: Qwen2.5-1.5B-Instruct
Adapter: LoRA SFT smoke adapter merged model
Eval: WebShop eval16
Max steps: 5

Result:
SFT-smoke improves action behavior over zero-shot.

Metrics:
- val/text/test_score: 0.625
- val/text/tool_call_count/mean: 0.0
- val/success_rate: 0.0625
- val/webshop_task_score: 0.0625
- episode/valid_action_ratio: 0.200
- response_length/mean: 91.600
- response_length/clip_ratio: 0.200
- prompt_length/mean: 1548.200
- timing_s/testing: 219.697
- timing_s/step: 249.620
- perf/max_memory_allocated_gb: 53.868
- perf/max_memory_reserved_gb: 66.607

Comparison with zero-shot eval16:
- zero-shot valid_action_ratio: 0.000
- SFT-smoke valid_action_ratio: 0.200
- zero-shot response_length/clip_ratio: 0.800
- SFT-smoke response_length/clip_ratio: 0.200
- zero-shot success_rate: 0.000
- SFT-smoke success_rate: 0.0625
- zero-shot val/text/test_score: 0.000
- SFT-smoke val/text/test_score: 0.625

Interpretation:
Even a 500-sample / 50-step LoRA SFT smoke run improves action-format behavior and reduces response clipping. This supports continuing to full SFT before SFT-only and SFT+GiGPO experiments.
