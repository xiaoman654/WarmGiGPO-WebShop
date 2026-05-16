# WebShop Step-level SFT Dataset Notes

Source:
data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl

Construction:
Each human trajectory is split into step-level samples:

instruction + previous actions + current observation -> next action

Output:
data/processed/sft_step_level/train.jsonl
data/processed/sft_step_level/valid.jsonl
data/processed/sft_step_level/stats.json

Status:
Generated successfully.

Stats:
{
  "input": "data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl",
  "num_samples": 15111,
  "num_train": 13600,
  "num_valid": 1511,
  "valid_ratio": 0.1,
  "seed": 42,
  "trajectory_count": 1571,
  "action_type_counts": {
    "click": 12914,
    "search": 2197
  },
  "avg_history_len": 9.789424922242075,
  "avg_observation_chars": 1190.5370921845013,
  "avg_target_action_chars": 48.55687909469923
}