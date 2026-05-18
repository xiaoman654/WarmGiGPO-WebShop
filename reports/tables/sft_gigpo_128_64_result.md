# SFT + GiGPO 128/64 Result

Date: 2026-05-18  
Model: Qwen2.5-1.5B-Instruct  
Warm-start: WebShop human-demo SFT, verl-aligned format  
Algorithm: GiGPO  
Environment: WebShop  
Train size: 128  
Eval size: 64  
Train batch size: 4  
Rollout n: 2  
Max episode steps: 5  
Training steps: 32  

## Final Validation

- val/text/test_score: 3.263888888888889
- val/success_rate: 0.328125
- val/webshop_task_score: 0.5605305284992785

## Rollout Behavior

- episode/valid_action_ratio: 1.000
- response_length/mean: 21.919
- response_length/max: 59.000
- response_length/min: 15.000
- response_length/clip_ratio: 0.000
- prompt_length/mean: 1166.730
- prompt_length/max: 2403.000
- prompt_length/clip_ratio: 0.000

## Training / Runtime

- timing_s/testing: 315.790
- timing_s/step: 337.398
- total runtime: about 33min36s
- perf/max_memory_allocated_gb: 65.877
- perf/max_memory_reserved_gb: 75.424
- perf/cpu_memory_used_gb: 132.939

## Notes

This is the strongest result so far. Compared with previous small and medium checks, SFT + GiGPO now achieves a clearly non-trivial WebShop success rate on eval64 while maintaining stable action formatting. The rollout remains short and well-formed, with valid_action_ratio at 1.0 and response_length/clip_ratio at 0.0.
