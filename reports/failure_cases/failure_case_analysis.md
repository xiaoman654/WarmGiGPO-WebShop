# Failure Case Analysis

Date: 2026-05-19  
Setting: WebShop train128/eval64, max_steps=5  
Methods compared: GiGPO vs SFT + GiGPO  

## Purpose

This analysis complements aggregate metrics by examining what kinds of behaviors differ between direct GiGPO and SFT + GiGPO.

The aggregate result shows:

| Method | success_rate | webshop_task_score | valid_action_ratio | response_len_mean | response_clip_ratio |
|---|---:|---:|---:|---:|---:|
| GiGPO | 0.0469 | 0.0511 | 0.974 | 73.487 | 0.026 |
| SFT + GiGPO | 0.3281 | 0.5605 | 1.000 | 21.919 | 0.000 |

The qualitative cases below explain why the gap appears.

## Observed Patterns

### Direct GiGPO

Direct GiGPO often produces long reasoning in the response and sometimes uses Chinese reasoning text despite an English WebShop prompt. The action is usually present, but the response spends many tokens on reasoning before the action. This matches the quantitative observation that direct GiGPO has much longer responses and non-zero clipping.

Representative failure patterns:

- **Overlong reasoning before action**: responses often include a full natural-language paragraph inside `<think>...</think>`.
- **Option or attribute mismatch**: the model may click an available option that only partially matches the instruction.
- **Item selection error**: the model clicks a product id from the page but not necessarily the correct item.
- **Search/query mismatch**: the generated search query sometimes keeps too many constraints or reformulates the task poorly.

Example direct GiGPO failure:

```text
Goal: find women's tops with color a01#black and size xx-large.
Response action: click[x-large]
Score: 0.0
Failure type: option error.
Reason: the requested size is xx-large, but the selected action is x-large.
```

Example direct GiGPO failure:

```text
Goal: find officially licensed machine-wash men's t-shirts with color silver, fit type youth, and size 2t.
Response action: click[youth]
Score: 0.0
Failure type: incomplete option selection / likely wrong progression.
Reason: selecting "youth" alone does not complete the full set of constraints and the trajectory receives zero score.
```

Example direct GiGPO failure:

```text
Goal: find wall art with color/theme "turtles and african american couple" and size 16x20 inch.
Response action: click[turtles and african american couple]
Score: 0.0
Failure type: attribute click without successful purchase.
Reason: the action is syntactically plausible but does not lead to a successful item purchase.
```

### SFT + GiGPO

SFT + GiGPO responses are much shorter and more regular. Most responses follow the same compact pattern:

```text
<think></think>
<action>...</action>
```

This is exactly the verl-aligned SFT target format. The improvement is not that every action is correct; rather, the model spends far less capacity on free-form reasoning and more consistently emits a parseable WebShop action.

Representative residual failure patterns:

- **Search still too broad or too literal**.
- **Wrong product click among plausible candidates**.
- **Wrong option click after reaching an item page**.
- **Navigation actions such as `click[< prev]` or `click[next >]` may still be overused.**

Example SFT + GiGPO failure:

```text
Goal: find machine-wash men's dress shirts with cotton spandex, classic fit, short sleeve, color patina green, size x-large.
Response action: search[patina green men's dress shirts with cotton spandex, classic fit, short sleeve, x-large, price lower than 60.00 dollars]
Score: 0.0
Failure type: search-stage failure.
Reason: the action format is clean, but the trajectory does not find and purchase the correct item within max_steps=5.
```

Example SFT + GiGPO failure:

```text
Response action: click[men]
Score: 0.0
Failure type: option or navigation error.
Reason: the action is short and parseable, but it selects an insufficient or wrong option for the task.
```

## Interpretation

The failure cases support the metric-level conclusion:

- SFT does not make the policy perfect.
- SFT mainly cleans up the action interface and reduces overlong generation.
- GiGPO then benefits from cleaner trajectories and obtains much higher final success.

In direct GiGPO, the model spends more tokens reasoning and often drifts into verbose explanations. In SFT + GiGPO, the model behaves more like a WebShop policy: it emits one short action at a time. This explains why SFT + GiGPO has lower response length, zero clipping, higher valid action ratio, and much higher final WebShop score.

## Next Steps

For a more systematic failure analysis, parse the rollout logs into structured JSONL and label 50-100 failed trajectories with:

```text
search error
item selection error
option error
premature buy
looping/navigation error
invalid action
overlong output
```

The helper script `scripts/analysis/extract_eval_cases.py` can be used to extract prompt/response/score cases from the logs before manual labeling.

