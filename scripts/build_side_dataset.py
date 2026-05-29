#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.side_dataset import build_side_dataset, side_dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    side_df = build_side_dataset(features)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    side_df.to_csv(output, index=False)

    summary = side_dataset_summary(side_df)
    if args.summary_output:
        summary_output = Path(args.summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_output, index=False)

    print(f"Wrote {len(side_df)} side rows to {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
