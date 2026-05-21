# Multi-turn SFT Follow-up Experiment

## Motivation

The current verl-aligned SFT data trains isolated step-level samples:

```text
user: current WebShop prompt with textual history
assistant: <think></think><action>{target_action}</action>
```

This format is effective for action-format alignment, but it also strongly
pushes the model toward empty reasoning. That makes the SFT policy behavior very
different from the base Qwen model, which often emits longer reasoning. This
weakens the interpretation of the KL reference ablation because the base
reference and SFT policy differ in both task behavior and output style.

The first follow-up is a bounded multi-turn SFT variant. It keeps the same
empty-think target for the current turn, but stores a short trajectory prefix as
alternating user/assistant messages. Historical assistant turns keep only the
final action, not `<think>`.

This follows the train-inference consistency principle: at inference time, the
agent should not rely on hidden reasoning from previous turns unless the runtime
also preserves that reasoning in context. It also avoids sequence-length
explosion from repeatedly nesting long WebShop prompts.

## Implemented Changes

- `scripts/data/build_webshop_sft_dataset.py`
  - new `--conversation-mode {single_turn,multi_turn}`
  - multi-turn mode emits bounded trajectory-prefix `messages`
  - new `--multiturn-history-window`, default `2`
  - new `--history-assistant`, default `action_only`
  - existing single-turn output remains the default
- `scripts/train/sft_lora.py`
  - supports arbitrary multi-turn `messages`
  - masks non-assistant tokens
  - supervises assistant turns for both two-turn and multi-turn rows
- `scripts/data/validate_webshop_sft_dataset.py`
  - validates even-length user/assistant message lists
  - reports `message_turns`
- server helper scripts
  - `scripts/data/build_webshop_sft_multiturn_verl.sh`
  - `scripts/train/run_qwen15b_lora_sft_verl_multiturn_500.sh`
  - `scripts/train/merge_sft_multiturn_adapters.sh`
  - `scripts/eval/run_qwen15b_sft_multiturn500_eval64.sh`
  - `scripts/rl/run_qwen15b_sft_multiturn500_gigpo_128_64.sh`

## First Experiment

Run only the 500-sample version first.

Compare against existing SFT-500:

| Variant | SFT data | Think target | SFT size | Eval |
|---|---|---|---:|---|
| baseline | single-turn step-level | empty | 500 | SFT-only eval64 + GiGPO 128/64 |
| candidate | bounded multi-turn, history action-only | empty current turn | 500 | SFT-only eval64 + GiGPO 128/64 |

Primary metrics:

- SFT-only `val/text/test_score`
- SFT-only `val/success_rate`
- SFT-only `val/webshop_task_score`
- SFT+GiGPO final metrics
- `episode/valid_action_ratio`
- `response_length/mean`
- `response_length/clip_ratio`
- `actor/kl_loss`

## Server Commands

```bash
cd /root/autodl-fs/WarmGiGPO-WebShop
git pull --rebase origin main

mkdir -p logs/train logs/eval logs/rl

echo "===== build multi-turn data ====="
bash scripts/data/build_webshop_sft_multiturn_verl.sh \
  2>&1 | tee logs/train/build_webshop_sft_multiturn_verl_$(date +%Y%m%d_%H%M%S).log

echo "===== train multi-turn SFT-500 ====="
bash scripts/train/run_qwen15b_lora_sft_verl_multiturn_500.sh \
  2>&1 | tee logs/train/qwen15b_lora_sft_verl_multiturn_500_$(date +%Y%m%d_%H%M%S).log

echo "===== merge multi-turn adapter ====="
bash scripts/train/merge_sft_multiturn_adapters.sh \
  2>&1 | tee logs/train/merge_sft_multiturn_adapters_$(date +%Y%m%d_%H%M%S).log

echo "===== SFT-only eval64 ====="
bash scripts/eval/run_qwen15b_sft_multiturn500_eval64.sh \
  2>&1 | tee logs/eval/qwen15b_sft_multiturn500_eval64_$(date +%Y%m%d_%H%M%S).log

echo "===== SFT+GiGPO 128/64 ====="
bash scripts/rl/run_qwen15b_sft_multiturn500_gigpo_128_64.sh \
  2>&1 | tee logs/rl/qwen15b_sft_multiturn500_gigpo_128_64_$(date +%Y%m%d_%H%M%S).log
```

## Interpretation

If bounded multi-turn SFT-500 beats single-turn SFT-500 after GiGPO, then
short trajectory context is useful even without generated reasoning.

If bounded multi-turn SFT-500 is similar or worse, the next better experiment
is not more empty-think multi-turn data. It is generated reasoning for the
current turn:

```text
<think>{DeepSeek-generated step reasoning}</think>
<action>{target_action}</action>
```

That would test whether reasoning supervision, not merely chat history, is the
missing ingredient.

Historical generated reasoning should not be kept by default. It would create a
train/inference mismatch unless the inference template also preserves historical
thinking.
