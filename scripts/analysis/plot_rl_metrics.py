#!/usr/bin/env python3
"""Parse verl-agent RL logs and plot report-ready metric curves.

The script expects logs produced by commands such as:

    logs/rl/qwen15b_sft_verl_gigpo_medium_128_64_*.log
    logs/rl/qwen15b_gigpo_medium_128_64_*.log

It writes one CSV per run and simple PNG curves for metrics that are present.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


STEP_RE = re.compile(r"step:(?P<step>\d+)\s+-\s+(?P<body>.*)")
PAIR_RE = re.compile(r"(?P<key>[A-Za-z0-9_()/.\- ]+):(?P<value>-?\d+(?:\.\d+)?)")
GROUP_RE = re.compile(r"Avg size of step-level group:\s*(?P<value>-?\d+(?:\.\d+)?)")

DEFAULT_METRICS = [
    "val/text/test_score",
    "val/success_rate",
    "val/webshop_task_score (not success_rate)",
    "episode/valid_action_ratio",
    "response_length/mean",
    "response_length/clip_ratio",
    "episode/webshop_task_score (not success_rate)",
    "episode/success_rate",
    "actor/kl_loss",
    "actor/entropy_loss",
    "critic/rewards/mean",
    "timing_s/step",
    "perf/max_memory_allocated_gb",
    "step_level_group_size",
]


def parse_log(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    pending_group_size: float | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            group_match = GROUP_RE.search(line)
            if group_match:
                pending_group_size = float(group_match.group("value"))
                continue

            step_match = STEP_RE.search(line)
            if not step_match:
                continue

            row: dict[str, float | int | str] = {
                "step": int(step_match.group("step")),
                "source_log": str(path),
            }
            if pending_group_size is not None:
                row["step_level_group_size"] = pending_group_size
                pending_group_size = None

            body = step_match.group("body")
            for item in body.split(" - "):
                pair = PAIR_RE.fullmatch(item.strip())
                if pair is None:
                    continue
                key = pair.group("key").strip()
                row[key] = float(pair.group("value"))

            rows.append(row)

    return rows


def write_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    keys = ["step", "source_log"]
    seen = set(keys)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(csv_paths: Iterable[Path], output_dir: Path, metrics: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        print(f"skip plotting because dependency is missing: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["run"] = csv_path.stem
        frames.append(df)

    if not frames:
        return

    merged = pd.concat(frames, ignore_index=True)
    for metric in metrics:
        if metric not in merged.columns:
            continue
        metric_df = merged.dropna(subset=[metric])
        if metric_df.empty:
            continue

        plt.figure(figsize=(8, 4.5))
        for run_name, group in metric_df.groupby("run"):
            group = group.sort_values("step")
            plt.plot(group["step"], group[metric], marker="o", label=run_name)
        plt.xlabel("global step")
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.tight_layout()
        safe_name = (
            metric.replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        plt.savefig(output_dir / f"{safe_name}.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="RL log files to parse")
    parser.add_argument("--out-dir", default="outputs/analysis/rl_metrics")
    parser.add_argument("--plot-dir", default="reports/figures/rl_metrics")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    csv_paths: list[Path] = []

    for log_arg in args.logs:
        log_path = Path(log_arg)
        rows = parse_log(log_path)
        out_path = out_dir / f"{log_path.stem}.csv"
        write_csv(rows, out_path)
        csv_paths.append(out_path)
        print(f"wrote {out_path} rows={len(rows)}")

    if not args.no_plots:
        plot_metrics(csv_paths, Path(args.plot_dir), DEFAULT_METRICS)
        print(f"wrote plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
