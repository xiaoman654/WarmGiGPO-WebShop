# Qwen2.5-1.5B A800 Tiny Smoke Test

Date: 2026-05-16
GPU: NVIDIA A800 80GB
Model: Qwen2.5-1.5B-Instruct
Environment: WebShop + verl-agent
Algorithm: GiGPO
Conda env: verl-agent-webshop

Config:
- train samples: 2
- val samples: 8
- train_batch_size: 1
- val_batch_size: 1
- max_prompt_length: 4096
- max_response_length: 128
- env.max_steps: 3
- env.rollout.n: 1
- total_epochs: 1
- test_freq: 1000

Result: SUCCESS

Observed metrics:
- Training Progress: 2/2 completed
- Max memory allocated: 53.868 GB
- Max memory reserved: 60.936 GB
- Step 1 time: 20.681 s
- Step 2 time: 68.464 s
- Final val/text/test_score: 0.0
- Final val/success_rate: 0.0
- Final val/webshop_task_score: 0.0
- Valid action ratio: 0.667 then 0.333
- Avg step-level group size: 1.5 then 1.0
- Episode length mean: 3.0
- Episode reward mean: 0.0

Notes:
- vLLM initialized successfully.
- CUDA graph capture completed successfully.
- WebShop worker ran successfully.
- GiGPO step-level grouping produced logs.
- Actor update, ref logprob, reward, and validation all completed.
- This confirms the minimal WebShop + verl-agent + GiGPO training chain works on one A800.
