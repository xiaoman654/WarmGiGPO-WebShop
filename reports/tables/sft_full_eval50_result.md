# SFT-full Eval50 Result

Date: 2026-05-16
Model: Qwen2.5-1.5B-Instruct
SFT: Full LoRA SFT, trajectory-level split
Eval: WebShop eval50
Max steps: 5

Metrics:
- val/text/test_score: 0.1606425702811245
- val/text/tool_call_count/mean: 0.0
- val/success_rate: 0.02
- val/webshop_task_score: 0.02
- episode/valid_action_ratio: 0.800
- response_length/mean: 101.200
- response_length/clip_ratio: 0.200
- prompt_length/mean: 1273.600
- timing_s/testing: 680.397
- timing_s/step: 712.670
- perf/max_memory_allocated_gb: 53.868
- perf/max_memory_reserved_gb: 65.443

Interpretation:
Full SFT substantially improves action validity and reduces response clipping. On eval50 it achieves non-zero success rate, suggesting that WebShop human-demo SFT provides a useful imitation prior, though final task success remains low and likely needs GiGPO RL.
