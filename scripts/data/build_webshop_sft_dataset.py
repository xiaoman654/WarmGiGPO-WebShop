#!/usr/bin/env python3
"""Build step-level SFT data from WebShop human demonstrations.

Input schema observed in WebShop `il_trajs_finalized_images.jsonl`:

{
  "actions": [...],
  "states": [...],
  "available_actions": [...],
  "actions_translate": [...],
  "action_idxs": [...],
  "images": [...]
}

Each state/action pair becomes:
instruction + previous actions + current observation -> next action.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ACTION_RE = re.compile(r"^\s*([a-zA-Z_ ]+)\[")


def extract_instruction(state: str) -> str:
    lines = state.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("instruction"):
            pieces = []
            for nxt in lines[idx + 1 :]:
                stripped = nxt.strip()
                if not stripped:
                    continue
                if (
                    stripped.startswith("[")
                    or stripped.startswith("Page ")
                    or stripped.startswith("Description:")
                    or stripped.startswith("Features:")
                    or stripped.startswith("Reviews:")
                ):
                    break
                pieces.append(stripped)
            return " ".join(pieces).strip()
    return ""


def action_type(action: str) -> str:
    match = ACTION_RE.match(action)
    if not match:
        return "unknown"
    return match.group(1).strip().lower().replace(" ", "_")


def make_prompt(instruction: str, history: list[str], observation: str) -> str:
    history_text = "\n".join(history) if history else "None"
    return (
        "You are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
        "Given the shopping instruction, previous actions, and current observation, output exactly one valid next action.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Action history:\n{history_text}\n\n"
        f"Current observation:\n{observation}\n\n"
        "Next action:"
    )


def iter_samples(input_path: Path) -> Iterable[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        for traj_id, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            states = obj.get("states") or []
            actions = obj.get("actions_translate") or obj.get("actions") or []
            raw_actions = obj.get("actions") or []
            available_actions = obj.get("available_actions") or []

            n_steps = min(len(states), len(actions))
            if n_steps == 0:
                continue

            instruction = extract_instruction(str(states[0]))
            for step_id in range(n_steps):
                target_action = str(actions[step_id]).strip()
                observation = str(states[step_id]).strip()
                if not target_action or not observation:
                    continue

                history = [str(action).strip() for action in actions[:step_id]]
                prompt = make_prompt(instruction, history, observation)
                sample_id = f"traj_{traj_id:05d}_step_{step_id:03d}"

                yield {
                    "sample_id": sample_id,
                    "trajectory_id": traj_id,
                    "step_id": step_id,
                    "instruction": instruction,
                    "history": history,
                    "observation": observation,
                    "available_actions": available_actions[step_id] if step_id < len(available_actions) else [],
                    "target_action": target_action,
                    "raw_action": str(raw_actions[step_id]).strip() if step_id < len(raw_actions) else target_action,
                    "action_type": action_type(target_action),
                    "prompt": prompt,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": target_action},
                    ],
                    "source": "webshop_human_demo_il_trajs_finalized_images",
                }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl",
    )
    parser.add_argument("--out-dir", default="data/processed/sft_step_level")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional debug cap.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Demo file not found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for sample in iter_samples(input_path):
        samples.append(sample)
        if args.max_samples and len(samples) >= args.max_samples:
            break

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_valid = max(1, int(len(samples) * args.valid_ratio)) if samples else 0
    valid = samples[:n_valid]
    train = samples[n_valid:]

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)

    stats = {
        "input": str(input_path),
        "num_samples": len(samples),
        "num_train": len(train),
        "num_valid": len(valid),
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "trajectory_count": len({s["trajectory_id"] for s in samples}),
        "action_type_counts": dict(Counter(s["action_type"] for s in samples).most_common()),
        "avg_history_len": sum(len(s["history"]) for s in samples) / max(1, len(samples)),
        "avg_observation_chars": sum(len(s["observation"]) for s in samples) / max(1, len(samples)),
        "avg_target_action_chars": sum(len(s["target_action"]) for s in samples) / max(1, len(samples)),
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote: {out_dir / 'train.jsonl'}")
    print(f"wrote: {out_dir / 'valid.jsonl'}")
    print(f"wrote: {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()

