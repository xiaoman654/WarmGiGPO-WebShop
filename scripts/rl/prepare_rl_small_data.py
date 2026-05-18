#!/usr/bin/env python3
"""Create small parquet splits for WebShop RL comparison runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-train", default="/root/data/verl-agent/text/train.parquet")
    parser.add_argument("--src-val", default="/root/data/verl-agent/text/test.parquet")
    parser.add_argument("--out-dir", default="/root/data/verl-agent/text_rl_small")
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.src_train).head(args.train_size)
    val = pd.read_parquet(args.src_val).head(args.val_size)

    train.to_parquet(out_dir / "train.parquet")
    val.to_parquet(out_dir / "test.parquet")

    print(f"wrote train: {out_dir / 'train.parquet'} rows={len(train)}")
    print(f"wrote val:   {out_dir / 'test.parquet'} rows={len(val)}")


if __name__ == "__main__":
    main()

