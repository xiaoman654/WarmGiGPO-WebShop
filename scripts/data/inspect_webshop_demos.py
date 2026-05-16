#!/usr/bin/env python3
"""Inspect WebShop human demonstration JSONL files.

This script is intentionally lightweight: it prints schema, counts, and a few
examples without loading the full file into memory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def describe(value: Any, name: str = "root", depth: int = 0, max_depth: int = 3) -> None:
    indent = "  " * depth
    print(f"{indent}{name}: {type(value).__name__}")
    if depth >= max_depth:
        print(f"{indent}repr: {repr(value)[:300]}")
        return

    if isinstance(value, dict):
        print(f"{indent}keys: {list(value.keys())}")
        for key, child in list(value.items())[:20]:
            describe(child, str(key), depth + 1, max_depth)
    elif isinstance(value, list):
        print(f"{indent}len: {len(value)}")
        if value:
            describe(value[0], "[0]", depth + 1, max_depth)
    else:
        print(f"{indent}repr: {repr(value)[:500]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl",
        help="Path to WebShop human demo JSONL.",
    )
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")

    print(f"path: {path}")
    print(f"size_mb: {path.stat().st_size / 1024 / 1024:.2f}")

    key_counts: Counter[str] = Counter()
    action_lens = []
    state_lens = []
    action_types: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            obj = json.loads(line)
            key_counts.update(obj.keys())
            actions = obj.get("actions_translate") or obj.get("actions") or []
            states = obj.get("states") or []
            action_lens.append(len(actions))
            state_lens.append(len(states))
            for action in actions:
                action_types.update([str(action).split("[", 1)[0].strip().lower()])

            if line_idx < args.examples:
                print("\n" + "=" * 80)
                print(f"example line: {line_idx}")
                describe(obj)

    print("\n" + "=" * 80)
    print(f"num_trajectories: {len(action_lens)}")
    print(f"keys: {dict(key_counts)}")
    print(f"action_steps_total: {sum(action_lens)}")
    print(f"action_len_min: {min(action_lens) if action_lens else 0}")
    print(f"action_len_max: {max(action_lens) if action_lens else 0}")
    print(f"action_len_mean: {sum(action_lens) / max(1, len(action_lens)):.2f}")
    print(f"state_len_min: {min(state_lens) if state_lens else 0}")
    print(f"state_len_max: {max(state_lens) if state_lens else 0}")
    print(f"state_len_mean: {sum(state_lens) / max(1, len(state_lens)):.2f}")
    print(f"action_types: {dict(action_types.most_common())}")


if __name__ == "__main__":
    main()

