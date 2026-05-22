#!/usr/bin/env python3
"""Export WebShop SFT rows into JSONL requests for external reasoning generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_actions(actions: Any) -> str:
    if not isinstance(actions, list):
        return str(actions)
    lines = []
    for idx, action in enumerate(actions, 1):
        lines.append(f"{idx}. {action}")
    return "\n".join(lines)


def make_generation_prompt(row: dict[str, Any], max_words: int, include_target_action: bool) -> str:
    history = row.get("history") or []
    available_actions = row.get("available_actions") or []
    base = (
        "You are generating a short rationale for the NEXT WebShop action at the current step.\n\n"
        "Your goal is to choose the single best next action strictly from the admissible actions, "
        "based only on the information provided below.\n\n"
        "Rules:\n"
        "- Use ONLY the Task, Previous actions, Current observation, and Admissible actions.\n"
        "- Focus on the CURRENT step only. Do not write a long plan or discuss future steps beyond what is needed for the next action.\n"
        "- Never invent prices, sizes, colors, ratings, or product names that are not explicitly shown in the Current observation.\n"
        "- Do NOT use any future information, hidden labels, ground truth, or target actions.\n"
        "- Compare the task requirements against the current observation. If a product clearly does NOT match "
        "(wrong category, wrong color, wrong size, price over budget, or missing a required attribute), say so briefly and prefer a better action.\n"
        "- If key information is still missing, prefer an information-gathering action (such as search, click, select, description, features, or reviews) over buying.\n"
        "- Only choose 'buy now' if the Current observation explicitly confirms ALL task-critical requirements and a 'buy now' action is available.\n"
        "- If previous actions show that similar searches already failed, try a better search strategy: shorten the query, use synonyms, or remove overly specific terms.\n"
        "- Do not repeat an action that has already proven ineffective (for example, repeating the exact same failed search query or revisiting the same clearly mismatched product). "
        "However, sequential browsing actions such as pagination may be reasonable if justified by the Current observation.\n"
        "- Mention at most 1-2 concrete cues from the Current observation or Admissible actions.\n"
        f"- Your reasoning must be concise, grounded, and no more than {max_words} words.\n"
        "- Output JSON only. Do not output markdown fences or any extra text.\n\n"
        f"Task:\n{row.get('instruction', '')}\n\n"
        "Previous actions:\n"
        f"{json.dumps(history, ensure_ascii=False) if history else '(none)'}\n\n"
        f"Current observation:\n{row.get('observation', '')}\n\n"
        "Admissible actions:\n"
        f"{format_actions(available_actions)}\n\n"
    )
    if include_target_action:
        base += (
            "Demonstrated next action:\n"
            f"{row.get('target_action', '')}\n\n"
            "Explain why this action is reasonable. Do not mention labels, ground truth, or demonstrations.\n"
        )
    else:
        base += (
            "Output JSON:\n"
            '{"think": "<short grounded rationale>", "chosen_action": "<exact action text copied from the admissible actions>"}\n'
        )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Base SFT JSONL file.")
    parser.add_argument("--output", required=True, help="Reasoning request JSONL.")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument(
        "--include-target-action",
        action="store_true",
        help="Include target_action in exported rows and prompt. Useful for rationalization, off by default.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset after loading the input rows, useful for sharding.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if args.start:
        rows = rows[args.start :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    requests = []
    for row in rows:
        request = {
            "sample_id": row.get("sample_id"),
            "trajectory_id": row.get("trajectory_id"),
            "step_id": row.get("step_id"),
            "instruction": row.get("instruction"),
            "history": row.get("history"),
            "observation": row.get("observation"),
            "available_actions": row.get("available_actions"),
            "conversation_mode": row.get("conversation_mode"),
            "prompt": make_generation_prompt(row, args.max_words, args.include_target_action),
        }
        if args.include_target_action:
            request["target_action"] = row.get("target_action")
        requests.append(request)

    write_jsonl(Path(args.output), requests)
    print(f"wrote requests: {args.output} rows={len(requests)}")


if __name__ == "__main__":
    main()
