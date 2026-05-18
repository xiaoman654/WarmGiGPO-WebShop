# RL Small Comparison

Date: 2026-05-18  
Model: Qwen2.5-1.5B-Instruct  
Environment: WebShop  
Algorithm: GiGPO  
Train size: 16  
Eval size: 16  
Rollout n: 2  
Max episode steps: 5  

## Original Qwen + GiGPO

Initial validation:
- val/text/test_score: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.020833333333333332

Final validation:
- val/text/test_score: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.020833333333333332

Behavior:
- final episode/valid_action_ratio: 1.000
- final response_length/mean: 84.722
- final response_length/clip_ratio: 0.000
- final timing_s/testing: 104.709
- final timing_s/step: 119.843
- perf/max_memory_allocated_gb: 61.844
- perf/max_memory_reserved_gb: 70.604

Notes:
- During training, valid_action_ratio was unstable and often low, including 0.200, 0.300, 0.450, and 0.600.
- Response length was much longer, often around 80-110 tokens.
- Several steps showed response clipping, with clip_ratio up to 0.500.

## SFT + GiGPO

Initial validation:
- val/text/test_score: 0.0
- val/success_rate: 0.0
- val/webshop_task_score: 0.12291666666666667

Final validation:
- val/text/test_score: 1.3157894736842106
- val/success_rate: 0.125
- val/webshop_task_score: 0.28005952380952376

Behavior:
- final episode/valid_action_ratio: 1.000
- final response_length/mean: 25.684
- final response_length/clip_ratio: 0.000
- final timing_s/testing: 77.934
- final timing_s/step: 91.483
- perf/max_memory_allocated_gb: 61.772
- perf/max_memory_reserved_gb: 71.930

Notes:
- SFT warm-start kept valid_action_ratio almost always at 1.000.
- Response length stayed short, mostly around 20-28 tokens.
- response_length/clip_ratio remained 0.000 throughout the run.
- Final validation achieved non-zero success_rate and substantially higher webshop_task_score than direct GiGPO.

## Preliminary Interpretation

This small run is not a final benchmark because train/eval sizes are small. However, it strongly supports the project's core hypothesis: WebShop human-demo SFT provides a useful imitation prior for GiGPO.

Compared with direct GiGPO, SFT + GiGPO improves:
- initial WebShop task score,
- final WebShop task score,
- success rate,
- valid action ratio,
- output length control,
- response clipping behavior.

The most important observation is that SFT warm-start improves the action distribution before RL, making GiGPO exploration less dominated by invalid or overlong outputs.
