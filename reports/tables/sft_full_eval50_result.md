# SFT-full Eval50 Result

Date: 2026-05-16
Model: Qwen2.5-1.5B-Instruct
SFT: Full LoRA SFT, trajectory-level split
Eval: WebShop eval50 via verl-agent main_ppo with val_before_train=True
Max steps: 5

Pure SFT-only initial validation:
- val/text/test_score: 0.0
- val/text/tool_call_count/mean: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.0

After one dummy GiGPO/PPO training step:
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
The initial validation should be treated as the strict SFT-only result and remains 0 on eval50. The final validation is not pure SFT-only because the script performs one training step before final validation. However, the immediate non-zero final result after one GiGPO/PPO step suggests that the SFT checkpoint is usable as an RL warm-start. This should be followed by a cleaner eval-only script and a prompt-aligned SFT target format.
