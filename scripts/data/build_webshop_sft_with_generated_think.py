#!/usr/bin/env python3
"""Merge externally generated reasoning into WebShop SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)


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


def extract_reasoning(row: dict[str, Any]) -> str:
    for key in ("think", "reasoning", "generated_think", "content", "response", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
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
    max_chars: int,
    require_all: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    merged = []
    missing = []
    for row in base_rows:
        sample_id = str(row.get("sample_id"))
        think = think_by_sample_id.get(sample_id, "")
        think = clean_reasoning(think, max_chars)
        if not think:
            missing.append(sample_id)
            if require_all:
                continue
            think = ""

        new_row = dict(row)
        new_row["generated_think"] = think
        new_row["think_source"] = "external_generated"
        new_row["assistant_target"] = format_assistant(str(row.get("target_action", "")), think)

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
    return merged, missing


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
    args = parser.parse_args()

    think_rows = load_jsonl(Path(args.think_file))
    think_by_sample_id = {}
    for row in think_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        think_by_sample_id[sample_id] = extract_reasoning(row)

    train_rows, missing_train = merge_rows(
        load_jsonl(Path(args.base_train)),
        think_by_sample_id,
        args.max_think_chars,
        args.require_all,
    )
    valid_rows, missing_valid = merge_rows(
        load_jsonl(Path(args.base_valid)),
        think_by_sample_id,
        args.max_think_chars,
        args.require_all,
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
        "max_think_chars": args.max_think_chars,
        "require_all": args.require_all,
        "avg_train_think_chars": sum(len(r.get("generated_think", "")) for r in train_rows) / max(1, len(train_rows)),
        "avg_valid_think_chars": sum(len(r.get("generated_think", "")) for r in valid_rows) / max(1, len(valid_rows)),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if missing_train or missing_valid:
        with (out_dir / "missing_think_sample_ids.txt").open("w", encoding="utf-8") as f:
            for sample_id in missing_train + missing_valid:
                f.write(sample_id + "\n")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote: {out_dir / 'train.jsonl'}")
    print(f"wrote: {out_dir / 'valid.jsonl'}")
    print(f"wrote: {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
