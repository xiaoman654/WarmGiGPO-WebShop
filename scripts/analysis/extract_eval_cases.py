#!/usr/bin/env python3
"""Extract prompt/response/score cases from verl-agent console logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PREFIX_RE = re.compile(r"^\(TaskRunner pid=\d+\)\s*")
SCORE_RE = re.compile(r"\[text\]\[score\]\s*(-?\d+(?:\.\d+)?)")


def clean_line(line: str) -> str:
    line = ANSI_RE.sub("", line).rstrip("\n")
    line = PREFIX_RE.sub("", line)
    return line


def extract_cases(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    section: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = clean_line(raw)

            if "[text][prompt]" in line:
                if current is not None:
                    cases.append(current)
                current = {
                    "source_log": str(path),
                    "prompt": line.split("[text][prompt]", 1)[1].strip(),
                    "response": "",
                    "score": None,
                }
                section = "prompt"
                continue

            if current is None:
                continue

            if "[text][response]" in line:
                current["response"] = line.split("[text][response]", 1)[1].strip()
                section = "response"
                continue

            score_match = SCORE_RE.search(line)
            if score_match:
                current["score"] = float(score_match.group(1))
                cases.append(current)
                current = None
                section = None
                continue

            if "test_gen_batch meta info" in line or "validation generation end" in line:
                continue

            if section == "prompt":
                current["prompt"] = str(current["prompt"]) + "\n" + line
            elif section == "response":
                current["response"] = str(current["response"]) + "\n" + line

    if current is not None:
        cases.append(current)
    return cases


def summarize_action(response: str) -> str:
    action_match = re.search(r"<action>(.*?)</action>", response, flags=re.DOTALL)
    if action_match:
        return " ".join(action_match.group(1).split())
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out-dir", default="outputs/analysis/eval_cases")
    parser.add_argument("--max-cases", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for log_arg in args.logs:
        path = Path(log_arg)
        cases = extract_cases(path)
        for case in cases:
            case["action"] = summarize_action(str(case.get("response", "")))

        output_jsonl = out_dir / f"{path.stem}.jsonl"
        with output_jsonl.open("w", encoding="utf-8") as f:
            for case in cases[: args.max_cases]:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

        success = sum(1 for case in cases if case.get("score") == 10.0)
        nonzero = sum(1 for case in cases if isinstance(case.get("score"), float) and case["score"] > 0)
        print(
            f"{path}: cases={len(cases)} nonzero={nonzero} success_score_10={success} "
            f"wrote={output_jsonl}"
        )


if __name__ == "__main__":
    main()

