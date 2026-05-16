# LoRA SFT Smoke Test Result

Date: 2026-05-16
Model: Qwen2.5-1.5B-Instruct
Method: LoRA SFT
Dataset: WebShop step-level human demonstrations
Train samples: 500
Valid samples: 100
Max steps: 50
Max sequence length: 2048
LoRA rank: 16
LoRA alpha: 32

Result: SUCCESS

Key metrics:
- Trainable params: 18,464,768
- Total params: 1,562,179,072
- Trainable ratio: 1.1820%
- Final train_loss: 0.5081
- Final eval_loss: 0.2822
- Train runtime: 135.27s
- Train steps per second: 0.37
- Final adapter: outputs/sft/qwen25_1p5b_webshop_lora_smoke/final_adapter

Loss trend:
- step 5: 2.2894
- step 10: 0.8266
- step 15: 0.3501
- step 20: 0.3407
- step 25: 0.2700
- step 50: 0.1516

Notes:
- LoRA SFT pipeline is functional.
- Loss decreases rapidly on 500-sample smoke data.
- Final adapter checkpoint is saved successfully.
- Warnings about Flash Attention, kernel version, tokenizer deprecation, and label_names are non-blocking for this smoke test.
