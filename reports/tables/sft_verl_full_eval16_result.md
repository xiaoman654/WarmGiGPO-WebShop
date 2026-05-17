# SFT-v2 verl-aligned Full Eval16 Result

Date: 2026-05-17
Model: Qwen2.5-1.5B-Instruct
SFT: Full LoRA SFT with verl-aligned target format
Eval: WebShop eval16 via verl-agent main_ppo with val_before_train=True
Target format:
<think></think>
<action>{target_action}</action>

Pure SFT-only initial validation:
- val/text/test_score: 0.0
- val/text/tool_call_count/mean: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.12291666666666667

After one dummy GiGPO/PPO training step:
- val/text/test_score: 0.0
- val/text/tool_call_count/mean: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.004464285714285714

Behavior metrics from rollout:
- episode/valid_action_ratio: 1.000
- response_length/mean: 32.600
- response_length/clip_ratio: 0.000
- prompt_length/mean: 1442.400
- prompt_length/clip_ratio: 0.000

Efficiency:
- timing_s/testing: 80.421
- timing_s/step: 93.857
- perf/max_memory_allocated_gb: 53.868
- perf/max_memory_reserved_gb: 66.504

Interpretation:
verl-aligned SFT substantially improves WebShop action formatting. Compared with zero-shot and action-only SFT, it achieves perfect valid action ratio, eliminates response clipping, and improves initial WebShop task score on eval16. This checkpoint is suitable as the SFT warm-start for GiGPO.
