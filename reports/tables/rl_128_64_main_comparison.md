# RL 128/64 Main Comparison

Date: 2026-05-18  
Model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
Algorithm: GiGPO  
Train size: 128  
Eval size: 64  
Train batch size: 4  
Rollout n: 2  
Max episode steps: 5  
Training steps: 32  

## Main Result

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 0.2893890675 | 0.046875 | 0.0511363636 | 0.974 | 73.487 | 0.026 |
| SFT + GiGPO | 3.2638888889 | 0.328125 | 0.5605305285 | 1.000 | 21.919 | 0.000 |

## Relative Improvement

- success_rate: 0.046875 -> 0.328125, about 7.0x
- webshop_task_score: 0.0511363636 -> 0.5605305285, about 11.0x
- response_length/mean: 73.487 -> 21.919
- response_length/clip_ratio: 0.026 -> 0.000

## Direct GiGPO Final Metrics

- val/text/test_score: 0.28938906752411575
- val/success_rate: 0.046875
- val/webshop_task_score: 0.05113636363636363
- episode/valid_action_ratio: 0.974
- response_length/mean: 73.487
- response_length/max: 128.000
- response_length/min: 32.000
- response_length/clip_ratio: 0.026
- prompt_length/mean: 1118.359
- prompt_length/max: 2238.000
- timing_s/testing: 413.634
- timing_s/step: 437.004
- perf/max_memory_allocated_gb: 66.071
- perf/max_memory_reserved_gb: 74.318

## SFT + GiGPO Final Metrics

- val/text/test_score: 3.263888888888889
- val/success_rate: 0.328125
- val/webshop_task_score: 0.5605305284992785
- episode/valid_action_ratio: 1.000
- response_length/mean: 21.919
- response_length/max: 59.000
- response_length/min: 15.000
- response_length/clip_ratio: 0.000
- prompt_length/mean: 1166.730
- prompt_length/max: 2403.000
- timing_s/testing: 315.790
- timing_s/step: 337.398
- perf/max_memory_allocated_gb: 65.877
- perf/max_memory_reserved_gb: 75.424

## Interpretation

The 128/64 comparison strongly supports the central hypothesis of the project. Under the same GiGPO configuration, the WebShop human-demo SFT warm-start substantially improves final WebShop performance, success rate, and action quality.

The largest behavioral difference is output control. Direct GiGPO still produces much longer responses and occasional clipped outputs, while SFT + GiGPO keeps responses short, valid, and consistently parseable. This suggests that SFT provides a useful imitation prior that improves the quality of GiGPO rollout data and reduces exploration waste from malformed or overlong actions.
