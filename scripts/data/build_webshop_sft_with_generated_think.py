#!/usr/bin/env python3
"""Merge externally generated reasoning into WebShop SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"(search|click|choose|buy)\[[^\[\]]+\]", flags=re.IGNORECASE)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)


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


def parse_json_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = FENCED_JSON_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_reasoning(row: dict[str, Any]) -> str:
    for key in ("think", "reasoning", "generated_think", "content", "response", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            parsed = parse_json_text(text)
            if parsed:
                nested = extract_reasoning(parsed)
                if nested:
                    return nested
            match = THINK_RE.search(text)
            if match:
                return match.group(1).strip()
            return text
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content.strip():
                match = THINK_RE.search(content)
                if match:
                    return match.group(1).strip()
                return content.strip()
    return ""


def extract_chosen_action(row: dict[str, Any]) -> str:
    for key in ("chosen_action", "action", "predicted_action", "final_action"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            match = ACTION_RE.search(value)
            return match.group(0).strip() if match else value.strip()
    for key in ("content", "response", "text"):
        value = row.get(key)
        if isinstance(value, str):
            parsed = parse_json_text(value)
            if parsed:
                nested = extract_chosen_action(parsed)
                if nested:
                    return nested
            match = ACTION_RE.search(value)
            if match:
                return match.group(0).strip()
    return ""


def clean_reasoning(text: str, max_chars: int) -> str:
    text = THINK_RE.sub(lambda m: m.group(1), text)
    text = re.sub(r"</?action>", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def format_assistant(action: str, think: str) -> str:
    return f"<think>{think}</think>\n<action>{action}</action>"


def merge_rows(
    base_rows: list[dict[str, Any]],
    think_by_sample_id: dict[str, str],
    chosen_by_sample_id: dict[str, str],
    max_chars: int,
    require_all: bool,
    require_action_match: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    merged = []
    missing = []
    mismatched = []
    for row in base_rows:
        sample_id = str(row.get("sample_id"))
        think = think_by_sample_id.get(sample_id, "")
        think = clean_reasoning(think, max_chars)
        chosen_action = chosen_by_sample_id.get(sample_id, "")
        target_action = str(row.get("target_action", "")).strip()
        if not think:
            missing.append(sample_id)
            if require_all:
                continue
            think = ""
        if require_action_match and chosen_action and chosen_action.lower() != target_action.lower():
            mismatched.append(sample_id)
            continue

        new_row = dict(row)
        new_row["generated_think"] = think
        new_row["think_source"] = "external_generated"
        if chosen_action:
            new_row["generated_chosen_action"] = chosen_action
            new_row["generated_action_matches_target"] = chosen_action.lower() == target_action.lower()
        new_row["assistant_target"] = format_assistant(target_action, think)

        messages = list(row.get("messages") or [])
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] = new_row["assistant_target"]
        else:
            messages = [
                {"role": "user", "content": str(row.get("prompt", ""))},
                {"role": "assistant", "content": new_row["assistant_target"]},
            ]
        new_row["messages"] = messages
        merged.append(new_row)
    return merged, missing, mismatched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--base-valid", required=True)
    parser.add_argument("--think-file", required=True, help="JSONL with sample_id and think/reasoning/content.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-think-chars", type=int, default=800)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Drop rows without generated think instead of falling back to empty think.",
    )
    parser.add_argument(
        "--require-action-match",
        action="store_true",
        help="Drop rows whose generated chosen_action disagrees with target_action. Rows without chosen_action are kept.",
    )
    args = parser.parse_args()

    think_rows = load_jsonl(Path(args.think_file))
    think_by_sample_id = {}
    chosen_by_sample_id = {}
    for row in think_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        think_by_sample_id[sample_id] = extract_reasoning(row)
        chosen_by_sample_id[sample_id] = extract_chosen_action(row)

    train_rows, missing_train, mismatch_train = merge_rows(
        load_jsonl(Path(args.base_train)),
        think_by_sample_id,
        chosen_by_sample_id,
        args.max_think_chars,
        args.require_all,
        args.require_action_match,
    )
    valid_rows, missing_valid, mismatch_valid = merge_rows(
        load_jsonl(Path(args.base_valid)),
        think_by_sample_id,
        chosen_by_sample_id,
        args.max_think_chars,
        args.require_all,
        args.require_action_match,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)

    stats = {
        "base_train": args.base_train,
        "base_valid": args.base_valid,
        "think_file": args.think_file,
        "num_think_rows": len(think_rows),
        "num_think_sample_ids": len(think_by_sample_id),
        "num_train": len(train_rows),
        "num_valid": len(valid_rows),
        "missing_train": len(missing_train),
        "missing_valid": len(missing_valid),
        "action_mismatch_train": len(mismatch_train),
        "action_mismatch_valid": len(mismatch_valid),
        "max_think_chars": args.max_think_chars,
        "require_all": args.require_all,
        "require_action_match": args.require_action_match,
        "avg_train_think_chars": sum(len(r.get("generated_think", "")) for r in train_rows) / max(1, len(train_rows)),
        "avg_valid_think_chars": sum(len(r.get("generated_think", "")) for r in valid_rows) / max(1, len(valid_rows)),
        "train_action_match_rate": sum(
            1 for r in train_rows if r.get("generated_action_matches_target") is True
        )
        / max(1, sum(1 for r in train_rows if "generated_action_matches_target" in r)),
        "valid_action_match_rate": sum(
            1 for r in valid_rows if r.get("generated_action_matches_target") is True
        )
        / max(1, sum(1 for r in valid_rows if "generated_action_matches_target" in r)),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if missing_train or missing_valid:
        with (out_dir / "missing_think_sample_ids.txt").open("w", encoding="utf-8") as f:
            for sample_id in missing_train + missing_valid:
                f.write(sample_id + "\n")

    if mismatch_train or mismatch_valid:
        with (out_dir / "action_mismatch_sample_ids.txt").open("w", encoding="utf-8") as f:
            for sample_id in mismatch_train + mismatch_valid:
                f.write(sample_id + "\n")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote: {out_dir / 'train.jsonl'}")
    print(f"wrote: {out_dir / 'valid.jsonl'}")
    print(f"wrote: {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
