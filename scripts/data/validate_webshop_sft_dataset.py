#!/usr/bin/env python3
"""Validate step-level WebShop SFT JSONL data before training."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ACTION_RE = re.compile(r"^\s*(search|click|choose|buy)\[[^\[\]]+\]\s*$", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def validate_rows(rows: list[dict[str, Any]], split: str, max_examples: int) -> dict[str, Any]:
    required = [
        "sample_id",
        "trajectory_id",
        "step_id",
        "instruction",
        "history",
        "observation",
        "target_action",
        "action_type",
        "prompt",
        "messages",
    ]

    errors = []
    warnings = []
    action_types: Counter[str] = Counter()
    traj_ids = set()
    sample_ids = set()
    prompt_lens = []
    obs_lens = []
    action_lens = []
    history_lens = []

    for idx, row in enumerate(rows):
        sample_id = row.get("sample_id", f"<missing:{idx}>")
        if sample_id in sample_ids:
            errors.append(f"{split}:{sample_id}: duplicate sample_id")
        sample_ids.add(sample_id)

        missing = [key for key in required if key not in row]
        if missing:
            errors.append(f"{split}:{sample_id}: missing keys {missing}")
            continue

        if not str(row["instruction"]).strip():
            errors.append(f"{split}:{sample_id}: empty instruction")
        if not str(row["observation"]).strip():
            errors.append(f"{split}:{sample_id}: empty observation")
        if not str(row["target_action"]).strip():
            errors.append(f"{split}:{sample_id}: empty target_action")
        if not isinstance(row["history"], list):
            errors.append(f"{split}:{sample_id}: history is not a list")

        target_action = str(row["target_action"]).strip()
        if not ACTION_RE.match(target_action):
            warnings.append(f"{split}:{sample_id}: suspicious action format: {target_action!r}")

        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            errors.append(f"{split}:{sample_id}: messages must be a 2-item list")
        else:
            if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                errors.append(f"{split}:{sample_id}: messages roles should be user/assistant")
            if messages[1].get("content") != target_action:
                errors.append(f"{split}:{sample_id}: assistant message does not equal target_action")

        try:
            traj_ids.add(int(row["trajectory_id"]))
        except Exception:
            errors.append(f"{split}:{sample_id}: invalid trajectory_id")

        action_types.update([str(row["action_type"])])
        prompt_lens.append(len(str(row["prompt"])))
        obs_lens.append(len(str(row["observation"])))
        action_lens.append(len(target_action))
        history_lens.append(len(row["history"]) if isinstance(row["history"], list) else 0)

    print(f"\n[{split}]")
    print(f"rows: {len(rows)}")
    print(f"trajectory_count: {len(traj_ids)}")
    print(f"action_type_counts: {dict(action_types.most_common())}")
    print(
        "prompt_chars: "
        f"mean={mean(prompt_lens):.1f} p50={pct(prompt_lens, 0.5)} "
        f"p90={pct(prompt_lens, 0.9)} max={max(prompt_lens) if prompt_lens else 0}"
    )
    print(
        "observation_chars: "
        f"mean={mean(obs_lens):.1f} p50={pct(obs_lens, 0.5)} "
        f"p90={pct(obs_lens, 0.9)} max={max(obs_lens) if obs_lens else 0}"
    )
    print(
        "history_len: "
        f"mean={mean(history_lens):.1f} p50={pct(history_lens, 0.5)} "
        f"p90={pct(history_lens, 0.9)} max={max(history_lens) if history_lens else 0}"
    )
    print(
        "target_action_chars: "
        f"mean={mean(action_lens):.1f} p50={pct(action_lens, 0.5)} "
        f"p90={pct(action_lens, 0.9)} max={max(action_lens) if action_lens else 0}"
    )

    for row in rows[:max_examples]:
        print("\nexample:")
        print(f"  sample_id: {row.get('sample_id')}")
        print(f"  instruction: {str(row.get('instruction'))[:180]}")
        print(f"  history_len: {len(row.get('history') or [])}")
        print(f"  target_action: {row.get('target_action')}")

    return {
        "errors": errors,
        "warnings": warnings,
        "trajectory_ids": traj_ids,
        "sample_ids": sample_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/sft_step_level/train.jsonl")
    parser.add_argument("--valid", default="data/processed/sft_step_level/valid.jsonl")
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-warnings", type=int, default=20)
    args = parser.parse_args()

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not valid_path.exists():
        raise FileNotFoundError(valid_path)

    train = load_jsonl(train_path)
    valid = load_jsonl(valid_path)

    train_result = validate_rows(train, "train", args.max_examples)
    valid_result = validate_rows(valid, "valid", args.max_examples)

    sample_overlap = train_result["sample_ids"] & valid_result["sample_ids"]
    traj_overlap = train_result["trajectory_ids"] & valid_result["trajectory_ids"]

    errors = train_result["errors"] + valid_result["errors"]
    warnings = train_result["warnings"] + valid_result["warnings"]

    if sample_overlap:
        errors.append(f"train/valid sample_id overlap: {len(sample_overlap)}")
    if traj_overlap:
        warnings.append(
            "train/valid trajectory_id overlap: "
            f"{len(traj_overlap)}. Prefer trajectory-level split for final SFT."
        )

    print("\n[summary]")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")

    if errors:
        print("\nErrors:")
        for err in errors[:50]:
            print(f"- {err}")
    if warnings:
        print("\nWarnings:")
        for warn in warnings[: args.max_warnings]:
            print(f"- {warn}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

