# SFT-GiGPO 项目整体规划

项目题目：SFT-GiGPO: Studying Imitation Warm-Start for Critic-Free LLM Agent Reinforcement Learning in WebShop

本项目固定采用 `Qwen2.5-1.5B-Instruct + WebShop + verl-agent/GiGPO`，研究 WebShop 人类示范轨迹 SFT warm-start 是否能提升 GiGPO 的训练效率、最终性能和动作质量。项目不扩展到 3B/7B，不引入额外环境，优先把一个清晰、可复现、可分析的小模型 Agentic RL 后训练流程做完整。

截至 2026-05-15，参考仓库为：

- WebShop: https://github.com/princeton-nlp/WebShop, HEAD `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`
- verl-agent: https://github.com/langfengQ/verl-agent, HEAD `796ed310287fa605c9292a0fce07a86d79fde05e`

## 1. 研究问题

核心问题：

> 在 WebShop 多轮网页交互环境中，WebShop human-demo SFT warm-start 是否能提升 GiGPO 的训练效率、最终性能和动作质量？

主对比：

```text
Qwen2.5-1.5B-Instruct -> GiGPO
Qwen2.5-1.5B-Instruct -> WebShop human-demo SFT -> GiGPO
```

需要回答的问题包括：

- SFT 是否让 GiGPO 更快达到较高 reward 或 success rate？
- SFT 是否提高最终 WebShop score 和 success rate？
- SFT 改善的是动作合法性、搜索质量、商品选择、选项选择，还是减少循环？
- SFT 是否降低探索多样性，导致策略过早收敛？
- SFT 对 GiGPO 的 anchor state grouping 和 step-level advantage 是否有正面影响？

## 2. 总体技术路线

完整 pipeline：

```text
准备 WebShop 环境和任务数据
-> 解析 WebShop human demonstrations
-> 构造 step-level SFT 数据
-> 对 Qwen2.5-1.5B-Instruct 做 LoRA SFT
-> 得到 SFT checkpoint
-> 直接从原始 instruct model 启动 GiGPO
-> 从 SFT checkpoint 启动 GiGPO
-> WebShop evaluation
-> 训练曲线、行为指标、失败类型、GiGPO 机制分析
```

建议先做最小闭环，再扩大规模：

```text
Phase 0: 服务器环境和上游仓库跑通
Phase 1: WebShop eval zero-shot 跑通
Phase 2: human-demo -> step-level SFT 数据构造跑通
Phase 3: SFT LoRA 训练和 SFT-only eval 跑通
Phase 4: GiGPO baseline 跑通
Phase 5: SFT+GiGPO 跑通
Phase 6: 指标统计、机制分析、失败案例分析和报告整理
```

## 3. 推荐仓库结构

建议最终项目仓库采用如下结构。`third_party/` 只在服务器端作为上游源码或子模块位置，本地规划阶段不要求完整运行。

```text
SFT-GiGPO-WebShop/
  README.md
  PROJECT_PLAN.md
  requirements.txt
  configs/
    env/
      webshop_env.yaml
    sft/
      qwen25_1p5b_lora_sft.yaml
    rl/
      gigpo_qwen25_1p5b.yaml
      sft_gigpo_qwen25_1p5b.yaml
      sft_gigpo_ref_original_ablation.yaml
    eval/
      webshop_eval.yaml
  data/
    raw/
      webshop/
      demonstrations/
    processed/
      sft_step_level/
      eval_tasks/
  scripts/
    setup/
      prepare_webshop.sh
      download_models.sh
    data/
      parse_webshop_demos.py
      build_sft_dataset.py
      validate_sft_dataset.py
    train/
      run_sft_lora.sh
      run_gigpo.sh
      run_sft_gigpo.sh
    eval/
      run_eval.sh
      collect_rollout_logs.py
    analysis/
      compute_main_metrics.py
      compute_behavior_metrics.py
      analyze_anchor_groups.py
      classify_failures.py
      plot_curves.py
  outputs/
    sft/
    rl/
    eval/
    analysis/
  reports/
    figures/
    tables/
    failure_cases/
  third_party/
    WebShop/
    verl-agent/
```

## 4. 环境与安装规划

本项目不在当前本地机器运行，正式运行应在 Linux GPU 服务器上完成。建议服务器环境：

```text
OS: Ubuntu 22.04 或相近 Linux 发行版
Python: 3.10
CUDA: 优先 12.4
GPU: 2-4 张 RTX 4090/5090，或 1-2 张 A100/H800
```

推荐安装顺序：

```bash
conda create -n sft-gigpo-webshop python=3.10 -y
conda activate sft-gigpo-webshop

git clone https://github.com/princeton-nlp/WebShop third_party/WebShop
git clone https://github.com/langfengQ/verl-agent third_party/verl-agent

# 先安装系统依赖。WebShop 检索组件通常需要 Java。
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk build-essential git-lfs

# PyTorch CUDA 版本建议按服务器 CUDA/driver 重新确认。
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 再安装项目依赖。
pip install -r requirements.txt

# verl-agent 建议 editable install。
cd third_party/verl-agent
pip install -e .
cd ../..

# WebShop 数据和检索 index 按上游 README/setup.sh 准备。
```

注意事项：

- WebShop 原始依赖较旧，verl-agent 训练栈较新，若二者安装产生版本覆盖，优先保证 verl-agent、vLLM、PyTorch、Transformers 兼容。
- WebShop 的 search/index 部分依赖 Java、pyserini、faiss，`requirements.txt` 只能覆盖 Python 包，系统依赖和大文件下载需要脚本单独处理。
- `flash-attn` 对 CUDA、PyTorch、编译工具链敏感，失败时可先用非 flash attention 跑小规模调试，再回到正式环境修复性能依赖。
- 对 4090/5090 服务器，优先使用 LoRA/QLoRA、gradient checkpointing、较小 rollout batch 和 group size，先确保 rollout/logprob/reward 稳定。

## 5. 数据构造方案

输入数据来自 WebShop human demonstrations。目标是把每条完整人类轨迹拆成 step-level supervised samples：

```text
input = instruction + history + current observation
output = next action
```

建议统一保存为 JSONL：

```json
{
  "sample_id": "demo_000001_step_03",
  "task_id": "demo_000001",
  "step_id": 3,
  "instruction": "I want ...",
  "history": [
    "search[portable folding desk fully assembled]",
    "click[...product title...]"
  ],
  "observation": "Current WebShop page text ...",
  "target_action": "click[Buy Now]",
  "action_type": "click",
  "source": "webshop_human_demo"
}
```

Prompt 模板建议固定为：

```text
You are a shopping assistant in WebShop.
Given the user instruction, previous actions, and current observation, output exactly one valid next action.

Instruction:
{instruction}

Action history:
{history}

Current observation:
{observation}

Next action:
```

数据构造时需要做的校验：

- `target_action` 必须是 WebShop 可解析动作，例如 `search[...]`、`click[...]`、`choose[...]`、`buy[...]`。
- 每个样本必须包含 instruction、observation、target_action。
- history 只包含当前 step 之前的动作，不能泄漏未来动作。
- 过滤空 observation、异常跳步、环境解析失败的 action。
- 统计 action_type 分布，避免所有样本被搜索或点击动作主导。

建议输出三个数据文件：

```text
data/processed/sft_step_level/train.jsonl
data/processed/sft_step_level/valid.jsonl
data/processed/sft_step_level/stats.json
```

## 6. SFT 训练方案

模型：

```text
Qwen2.5-1.5B-Instruct
```

训练方式：

```text
LoRA SFT
```

建议初始超参：

```text
max_seq_length: 4096
learning_rate: 1e-4 或 2e-4
num_train_epochs: 2-3
per_device_train_batch_size: 1-2
gradient_accumulation_steps: 8-16
lora_rank: 16 或 32
lora_alpha: 32 或 64
lora_dropout: 0.05
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
bf16: true
gradient_checkpointing: true
```

SFT checkpoint 输出：

```text
outputs/sft/qwen25_1p5b_webshop_lora/
```

SFT 阶段需要记录：

- train/valid loss
- exact action string match
- action type accuracy
- valid action rate
- search/click/choose/buy 的分类准确率
- 少量 held-out demo 的 trajectory replay 质量

SFT 的目标不是最大化 WebShop reward，而是让模型学会动作格式、基本状态理解和合理下一步决策。

## 7. RL 训练方案

RL 框架使用 verl-agent 中的 GiGPO 实现。主实验包括：

```text
A. Prompting / Zero-shot
B. SFT only
C. GiGPO
D. SFT + GiGPO
```

GiGPO baseline：

```text
init_policy = Qwen2.5-1.5B-Instruct
reference_policy = Qwen2.5-1.5B-Instruct
algorithm = GiGPO
```

SFT+GiGPO 主实验：

```text
init_policy = SFT checkpoint
reference_policy = SFT checkpoint
algorithm = GiGPO
```

关键 ablation：

```text
D1. init_policy = SFT checkpoint, reference_policy = SFT checkpoint
D2. init_policy = SFT checkpoint, reference_policy = original instruct model
```

推荐调试规模：

```text
train tasks: 500-1000
eval tasks: 100
group size: 2 或 4
iterations: 20-50
max episode steps: 8-10
rollout temperature: 0.7-1.0
```

推荐正式规模：

```text
train tasks: 1000-3000
eval tasks: 300-500
group size: 4，资源允许再试 8
iterations: 100-200
max episode steps: 10-15
rollout temperature: 0.7-1.0
```

RL 日志必须保存完整 episode 信息：

```json
{
  "run_id": "sft_gigpo_seed_1",
  "iteration": 12,
  "task_id": "task_00042",
  "episode_id": "task_00042_rollout_03",
  "step": 4,
  "instruction": "...",
  "observation_hash": "...",
  "anchor_state_id": "...",
  "action": "click[...]",
  "action_type": "click",
  "valid_action": true,
  "reward": 0.0,
  "final_score": null,
  "done": false,
  "logprob": -1.23,
  "ref_logprob": -1.31,
  "kl": 0.08,
  "trajectory_advantage": 0.42,
  "step_advantage": 0.17
}
```

这些字段是后续行为分析和 anchor state grouping 分析的基础。

## 8. 评价指标

主性能指标：

```text
WebShop Score
Success Rate
Average Reward
```

训练效率指标：

```text
Reward curve
Success curve
达到指定 success rate 所需 iterations
Reward curve AUC
Success curve AUC
```

行为分析指标：

```text
Average episode length
Average search count
Average clicked item count
Invalid action rate
Premature buy rate
Repeated state rate
Repeated action rate
Option selection accuracy
Search query length
Search query diversity
Buy action precision
```

GiGPO 机制指标：

```text
anchor state group 数量
平均 group size
group size = 1 的比例
group size >= 4 的比例
同一 anchor state 下 action diversity
step-level advantage 均值、方差和分位数
positive/negative step advantage 比例
trajectory-level advantage 与 step-level advantage 的相关性
```

## 9. 失败类型分析

失败轨迹建议分类为：

```text
1. Search error: 搜索 query 不合适，导致候选商品召回失败。
2. Item selection error: 候选中存在正确商品，但点击错误商品。
3. Option error: 商品正确，但颜色、尺寸、规格等选项错误。
4. Premature buy: 还没有充分确认商品或选项就购买。
5. Looping: 反复搜索、反复点击、反复回到同一状态。
6. Invalid action: 模型输出环境无法解析的动作。
7. Memory failure: 曾看到正确商品或选项，但后续动作忘记该信息。
```

建议每个实验组抽样 50-100 条失败轨迹，由脚本初筛、人工复核。最终报告重点比较：

```text
GiGPO vs SFT+GiGPO
```

需要特别观察 SFT 是否减少 invalid action 和 looping，同时是否增加 premature buy 或降低探索。

## 10. 实验矩阵

主实验：

| ID | 初始化 | SFT | RL | Reference | 目的 |
| --- | --- | --- | --- | --- | --- |
| A | Qwen2.5-1.5B-Instruct | no | no | none | zero-shot baseline |
| B | Qwen2.5-1.5B-Instruct | yes | no | none | SFT-only baseline |
| C | Qwen2.5-1.5B-Instruct | no | GiGPO | original model | RL baseline |
| D | SFT checkpoint | yes | GiGPO | SFT checkpoint | main method |

可选 ablation：

| ID | 初始化 | RL | Reference | 目的 |
| --- | --- | --- | --- | --- |
| D1 | SFT checkpoint | GiGPO | SFT checkpoint | 主实验 reference 设置 |
| D2 | SFT checkpoint | GiGPO | original model | 分析 KL reference 对 warm-start 的影响 |
| D3 | SFT checkpoint | GiGPO | SFT checkpoint | 降低 rollout temperature，观察探索不足 |
| D4 | SFT checkpoint | GiGPO | SFT checkpoint | 增大 group size，观察 anchor grouping 质量 |

每个主要实验至少建议 2-3 个随机种子，资源紧张时优先保证 C 和 D 多种子。

## 11. 结果表格模板

主结果表：

| Method | WebShop Score | Success Rate | Avg Reward | Avg Steps | Invalid Action | Premature Buy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompting | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT only | TBD | TBD | TBD | TBD | TBD | TBD |
| GiGPO | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT+GiGPO | TBD | TBD | TBD | TBD | TBD | TBD |

训练效率表：

| Method | Best Reward | Best Success | Iter to Target Success | Reward AUC | Success AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| GiGPO | TBD | TBD | TBD | TBD | TBD |
| SFT+GiGPO | TBD | TBD | TBD | TBD | TBD |

机制分析表：

| Method | Avg Group Size | Group Size = 1 | Action Diversity | Step Adv Std | Repeated State |
| --- | ---: | ---: | ---: | ---: | ---: |
| GiGPO | TBD | TBD | TBD | TBD | TBD |
| SFT+GiGPO | TBD | TBD | TBD | TBD | TBD |

## 12. 风险与对策

环境风险：

- WebShop 原始依赖旧，verl-agent 依赖新，可能出现版本冲突。
- 对策：使用独立 conda 环境；按 WebShop -> PyTorch/vLLM/verl-agent 的顺序安装；把可复现安装命令写入 `scripts/setup/prepare_webshop.sh`。

训练不稳定：

- 小模型 RL 可能出现 invalid action、reward 长期为零、KL 爆炸。
- 对策：先验证 SFT-only valid action rate；GiGPO 调试阶段降低任务数、缩短 episode、减小 learning rate；固定 prompt 和 action parser。

SFT 过拟合：

- SFT 可能学会演示风格但降低探索多样性。
- 对策：保留 validation demos；记录 action diversity；做 reference ablation 和 temperature ablation。

评估噪声：

- WebShop 任务差异大，单次 eval 不稳定。
- 对策：固定 eval task split；每个主实验多 seed；报告均值和标准差。

机制日志缺失：

- 如果早期没有记录 anchor state、step advantage、action diversity，后续无法补分析。
- 对策：RL 第一个可运行版本就加入结构化 rollout 日志，不只依赖 wandb 曲线。

## 13. 里程碑

建议按 4-6 周安排：

```text
Week 1:
  服务器环境、WebShop、verl-agent 安装；
  zero-shot eval 跑通；
  明确 task split。

Week 2:
  human-demo parser；
  step-level SFT dataset；
  dataset validation 和统计。

Week 3:
  LoRA SFT；
  SFT-only eval；
  修正 prompt/action format。

Week 4:
  GiGPO baseline；
  SFT+GiGPO 主实验；
  保存完整 rollout 日志。

Week 5:
  多 seed 或扩大正式规模；
  reference ablation；
  行为指标和 anchor grouping 分析。

Week 6:
  失败案例复核；
  结果表格、曲线、报告和 README。
```

## 14. 最终交付物

必须交付：

- 可运行代码仓库
- WebShop human-demo 解析脚本
- step-level SFT 数据构造脚本
- SFT LoRA 训练脚本
- GiGPO 和 SFT+GiGPO 训练脚本
- WebShop evaluation 脚本
- 主结果表格
- reward/success 曲线
- 行为分析图表
- anchor state grouping 机制分析
- 失败案例分析
- README 或项目报告

最终 README 建议结构：

```text
Background
Research Question
Method
Data Construction
Training Pipeline
Experiments
Results
Behavior Analysis
GiGPO Mechanism Analysis
Failure Analysis
Limitations
Future Work
```

## 15. 第一版执行优先级

第一版只追求完整闭环：

```text
1. 跑通 WebShop eval
2. 构造 SFT JSONL
3. 跑通 Qwen2.5-1.5B-Instruct LoRA SFT
4. 跑通 SFT-only eval
5. 跑通 GiGPO 20-50 iterations
6. 跑通 SFT+GiGPO 20-50 iterations
7. 输出最小结果表和两张曲线
```

在这个闭环稳定之前，不扩模型、不加环境、不做复杂 ablation。

