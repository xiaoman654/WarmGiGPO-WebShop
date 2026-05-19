# SFT-GiGPO: 基于 WebShop 的小模型 Agentic RL 后训练研究

## 摘要

本项目研究 WebShop 多轮网页交互环境中，WebShop human-demo SFT warm-start 是否能够提升 GiGPO 的训练效果和 rollout 质量。实验固定使用 Qwen2.5-1.5B-Instruct，不扩展到更大模型，重点考察小模型在 WebShop 中经过 imitation warm-start 后，是否能更稳定地进入 critic-free Agentic RL 训练。

当前主实验结果显示，在 train128/eval64 设置下，相比直接 GiGPO，SFT + GiGPO 显著提升了 WebShop 成功率和任务分数：

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 0.2894 | 0.0469 | 0.0511 | 0.974 | 73.487 | 0.026 |
| SFT + GiGPO | 3.2639 | 0.3281 | 0.5605 | 1.000 | 21.919 | 0.000 |

SFT + GiGPO 将 success rate 从 0.0469 提升到 0.3281，约 7.0 倍；将 WebShop task score 从 0.0511 提升到 0.5605，约 11.0 倍。行为指标也显示，SFT warm-start 使模型输出更短、更稳定、更容易被 WebShop 环境解析。

## 研究问题

本项目的核心问题是：

> 在 WebShop 这类多轮网页交互 Agent 环境中，SFT warm-start 是否能够提升 GiGPO 的训练效率、最终性能和动作质量？

具体比较：

```text
Qwen2.5-1.5B-Instruct -> GiGPO
Qwen2.5-1.5B-Instruct -> WebShop human-demo SFT -> GiGPO
```

本项目不是单纯复现 GiGPO，而是研究 imitation prior 对 critic-free Agentic RL 的影响：它是否减少无效探索、改善动作格式、提高 rollout 质量，并进一步改善 GiGPO 的训练信号。

## 方法

### 模型与环境

- Base model: Qwen2.5-1.5B-Instruct
- Environment: WebShop
- RL framework: verl-agent
- RL algorithm: GiGPO
- SFT method: LoRA
- RL warm-start: merged SFT checkpoint

### WebShop SFT 数据构造

WebShop human demonstration 数据来自 `il_trajs_finalized_images.jsonl`，共 1571 条轨迹。每条轨迹包含：

```text
actions
states
available_actions
actions_translate
action_idxs
images
```

我们将完整轨迹拆成 step-level SFT 样本：

```text
instruction + action history + current observation + admissible actions -> next action
```

由于 verl-agent 的 WebShop prompt 要求模型输出：

```text
<think>...</think>
<action>...</action>
```

而 human demonstration 中没有真实 think 数据，因此本项目采用空 think 的 verl-aligned target：

```text
<think></think>
<action>{target_action}</action>
```

这样做避免了用大模型伪造 reasoning，同时保证 SFT 输出格式与 verl-agent 的 action parser 一致。

最终 SFT 数据规模：

| Split | Samples | Trajectories |
|---|---:|---:|
| Train | 13625 | 1414 |
| Valid | 1486 | 157 |

数据验证结果：

- errors: 0
- warnings: 36
- warnings 主要来自商品标题中包含 `[` 或 `]`，属于 WebShop 商品名本身的格式复杂性，不影响主要训练。

## 实验设置

### 已完成实验

1. WebShop environment smoke test
2. Zero-shot / prompting baseline
3. Human-demo SFT 数据构造与验证
4. verl-aligned LoRA SFT smoke training
5. verl-aligned LoRA SFT full training
6. SFT-only WebShop eval
7. GiGPO smoke / small run
8. SFT + GiGPO smoke / small run
9. GiGPO vs SFT + GiGPO train128/eval64 主实验

### 主实验配置

| Item | Value |
|---|---|
| train size | 128 |
| eval size | 64 |
| train batch size | 4 |
| rollout n | 2 |
| max episode steps | 5 |
| training steps | 32 |
| rollout engine | vLLM |
| GPUs | 1 x A800 80GB |

## 主要结果

### 128/64 主对比

| Method | val/text/test_score | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | 0.2894 | 0.0469 | 0.0511 | 0.974 | 73.487 | 0.026 |
| SFT + GiGPO | 3.2639 | 0.3281 | 0.5605 | 1.000 | 21.919 | 0.000 |

### 结果解释

SFT + GiGPO 在三个层面显著优于 direct GiGPO：

1. 主性能指标更高：
   - success rate 提升约 7.0 倍。
   - WebShop task score 提升约 11.0 倍。

2. 动作质量更稳定：
   - valid_action_ratio 达到 1.000。
   - direct GiGPO 虽然最终也接近合法输出，但训练过程中 action validity 更不稳定。

3. 输出长度更可控：
   - SFT + GiGPO 的 response length mean 为 21.919。
   - direct GiGPO 的 response length mean 为 73.487。
   - SFT + GiGPO 的 response clipping 为 0。

这说明 SFT warm-start 的主要作用不是简单提高初始 reward，而是改善 rollout 分布，使模型更容易输出 WebShop 可解析的短动作。

## 曲线分析

已生成曲线位于：

```text
reports/figures/rl_metrics/
```

推荐主文展示：

- `val_success_rate.png`
- `val_webshop_task_score_not_success_rate.png`
- `episode_valid_action_ratio.png`
- `response_length_mean.png`
- `response_length_clip_ratio.png`
- `step_level_group_size.png`

这些曲线显示：

- SFT + GiGPO 的 validation performance 随训练明显上升。
- direct GiGPO 的 validation score 较低且提升不稳定。
- SFT + GiGPO 的 response length 长期保持在 20-30 tokens。
- direct GiGPO 的 response length 长期在 70-100 tokens。
- SFT + GiGPO 基本没有 response clipping。
- step-level group size 大多在 1.5-2.3 之间，符合 `env.rollout.n=2` 的设置。

## GiGPO 机制视角

GiGPO 的关键是 anchor state grouping 和 step-level advantage。SFT warm-start 可能有两种相反影响：

- 正向影响：减少无效动作和过长输出，使同一状态下的 action comparison 更干净。
- 负向影响：降低探索多样性，使策略过早集中。

当前实验结果显示，正向影响占主导。SFT + GiGPO 在保持较高合法动作率的同时，显著提升了 final score 和 success rate。这说明 imitation prior 没有在当前设置下阻碍 GiGPO 学习，反而提升了 rollout 数据质量。

## 资源与效率

主实验在 1 张 A800 80GB 上完成。

SFT + GiGPO 128/64：

- runtime: about 33min36s
- max_memory_allocated_gb: 65.877
- max_memory_reserved_gb: 75.424

Direct GiGPO 128/64：

- max_memory_allocated_gb: 66.071
- max_memory_reserved_gb: 74.318

训练中最耗时的是 validation step。普通训练 step 约 20-35 秒，但带 validation 的 step 会达到数百秒。因此后续扩大规模时，控制 `test_freq` 和 eval size 是降低成本的关键。

## 局限性

当前结果已经支持主结论，但仍有几个限制：

1. 训练和评估规模仍偏小：
   - 主实验为 train128/eval64，不是完整 WebShop benchmark。

2. max episode steps 较短：
   - 当前使用 max_steps=5，正式 WebShop 任务可能需要 10-15 steps。

3. seed 数量有限：
   - 当前主实验主要是单 seed。

4. 失败案例分析尚未系统完成：
   - 目前主要分析 aggregate metrics 和训练曲线。

5. KL reference ablation 尚未完成：
   - 后续可比较 reference=SFT checkpoint 和 reference=original Qwen。

## 后续扩展

后续扩展不会影响当前报告主结论，可以作为 additional experiments：

1. SFT 数据量 ablation：
   - SFT-500
   - SFT-2k
   - SFT-full

2. KL reference ablation：
   - SFT + GiGPO, reference = SFT checkpoint
   - SFT + GiGPO, reference = original Qwen

3. 失败类型分析：
   - search error
   - item selection error
   - option error
   - premature buy
   - looping
   - invalid action
   - memory failure

4. 更大 WebShop 设置：
   - train256/eval100
   - max_steps=8 或 10

## 当前结论

在当前 WebShop train128/eval64 设置下，WebShop human-demo SFT warm-start 明显提升了 Qwen2.5-1.5B-Instruct 的 GiGPO 后训练效果。SFT + GiGPO 不仅取得更高的 success rate 和 WebShop task score，还显著改善了 rollout 行为，包括动作合法性、输出长度和 response clipping。

这表明，在 critic-free Agentic RL 中，SFT imitation prior 可以作为有效的 warm-start，降低早期探索难度，并提高 GiGPO 的训练信号质量。

