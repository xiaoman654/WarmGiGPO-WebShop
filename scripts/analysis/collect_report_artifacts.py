#!/usr/bin/env python3
"""Collect key experiment artifacts into report-friendly files.

This script does not require model execution. It scans existing notes, logs, and
dataset statistics and writes a compact Markdown summary plus a machine-readable
JSON file under `reports/`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(r"([A-Za-z0-9_/\-() ]+):(-?\d+(?:\.\d+)?)")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_final_validation_metrics(text: str) -> dict[str, float]:
    result = {}
    for key in [
        "val/text/test_score",
        "val/text/tool_call_count/mean",
        "val/success_rate",
        "val/webshop_task_score (not success_rate)",
    ]:
        pattern = re.escape(key) + r"':?\s*([0-9eE+\-.]+)"
        matches = re.findall(pattern, text)
        if matches:
            try:
                result[key] = float(matches[-1])
            except ValueError:
                pass
    return result


def parse_step_metrics(text: str) -> dict[str, float]:
    keys = [
        "episode/valid_action_ratio",
        "response_length/clip_ratio",
        "prompt_length/mean",
        "response_length/mean",
        "timing_s/testing",
        "timing_s/step",
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
    ]
    result = {}
    for key in keys:
        pattern = re.escape(key) + r":(-?[0-9.]+)"
        matches = re.findall(pattern, text)
        if matches:
            result[key] = float(matches[-1])
    return result


def latest_file(pattern: str) -> Path | None:
    files = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def collect_sample_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "sample_id": obj.get("sample_id"),
                    "trajectory_id": obj.get("trajectory_id"),
                    "step_id": obj.get("step_id"),
                    "instruction": obj.get("instruction"),
                    "history_len": len(obj.get("history") or []),
                    "target_action": obj.get("target_action"),
                    "action_type": obj.get("action_type"),
                    "observation_prefix": str(obj.get("observation", ""))[:300],
                }
            )
            if len(rows) >= limit:
                break
    return rows


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# WarmGiGPO-WebShop Report Artifacts",
        "",
        "## Environment And Smoke Test",
        "",
    ]

    smoke = report.get("smoke", {})
    if smoke:
        lines.extend(
            [
                f"- Summary file: `{smoke.get('summary_file', '')}`",
                f"- Result: {smoke.get('result', 'recorded')}",
                "",
            ]
        )
    else:
        lines.append("- No smoke summary found.\n")

    lines.extend(["## Zero-shot Baseline", ""])
    zero = report.get("zero_shot", {})
    if zero:
        lines.append(f"- Notes file: `{zero.get('notes_file', '')}`")
        lines.append(f"- Latest log: `{zero.get('latest_log', '')}`")
        for key, value in zero.get("metrics", {}).items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    else:
        lines.append("- No zero-shot baseline notes/logs found.\n")

    lines.extend(["## SFT Dataset", ""])
    dataset = report.get("sft_dataset", {})
    stats = dataset.get("stats", {})
    if stats:
        lines.append(f"- Stats file: `{dataset.get('stats_file', '')}`")
        for key in [
            "num_samples",
            "num_train",
            "num_valid",
            "trajectory_count",
            "avg_history_len",
            "avg_observation_chars",
            "avg_target_action_chars",
        ]:
            if key in stats:
                lines.append(f"- {key}: {stats[key]}")
        lines.append(f"- action_type_counts: `{stats.get('action_type_counts', {})}`")
        lines.append("")
    else:
        lines.append("- No SFT dataset stats found.\n")

    samples = dataset.get("examples", [])
    if samples:
        lines.extend(["### SFT Sample Examples", ""])
        for sample in samples:
            lines.extend(
                [
                    f"- sample_id: `{sample.get('sample_id')}`",
                    f"  - instruction: {sample.get('instruction')}",
                    f"  - history_len: {sample.get('history_len')}",
                    f"  - target_action: `{sample.get('target_action')}`",
                    f"  - action_type: `{sample.get('action_type')}`",
                ]
            )
        lines.append("")

    validation = report.get("sft_validation", {})
    lines.extend(["## SFT Dataset Validation", ""])
    if validation:
        lines.append(f"- Notes file: `{validation.get('notes_file', '')}`")
        lines.append(f"- errors: {validation.get('errors')}")
        lines.append(f"- warnings: {validation.get('warnings')}")
    else:
        lines.append("- No SFT validation notes found.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/generated")
    parser.add_argument("--sample-limit", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke_summary = Path("logs/smoke/qwen15b_a800_tiny_smoke_summary.md")
    zero_notes = Path("outputs/eval/zero_shot_eval16_notes.md")
    zero_log = latest_file("logs/eval/qwen15b_zero_shot_eval16_*.log")
    sft_stats = Path("data/processed/sft_step_level/stats.json")
    sft_train = Path("data/processed/sft_step_level/train.jsonl")
    sft_validation_notes = Path("reports/tables/sft_dataset_validation_notes.md")

    zero_text = read_text(zero_notes) + "\n" + (read_text(zero_log) if zero_log else "")
    zero_metrics = parse_final_validation_metrics(zero_text)
    zero_metrics.update(parse_step_metrics(zero_text))

    validation_text = read_text(sft_validation_notes)
    error_match = re.search(r"errors:\s*(\d+)", validation_text)
    warning_match = re.search(r"warnings:\s*(\d+)", validation_text)

    report = {
        "smoke": {
            "summary_file": str(smoke_summary) if smoke_summary.exists() else None,
            "result": "success" if "SUCCESS" in read_text(smoke_summary) else None,
        },
        "zero_shot": {
            "notes_file": str(zero_notes) if zero_notes.exists() else None,
            "latest_log": str(zero_log) if zero_log else None,
            "metrics": zero_metrics,
        },
        "sft_dataset": {
            "stats_file": str(sft_stats) if sft_stats.exists() else None,
            "stats": load_json(sft_stats),
            "examples": collect_sample_rows(sft_train, args.sample_limit),
        },
        "sft_validation": {
            "notes_file": str(sft_validation_notes) if sft_validation_notes.exists() else None,
            "errors": int(error_match.group(1)) if error_match else None,
            "warnings": int(warning_match.group(1)) if warning_match else None,
        },
    }

    json_path = out_dir / "report_artifacts.json"
    md_path = out_dir / "report_artifacts.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()

