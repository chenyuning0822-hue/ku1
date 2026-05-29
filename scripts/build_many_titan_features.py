#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.titan import build_titan_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract-root", default="data/raw/extracted")
    parser.add_argument("--table-label", default="亚盘")
    parser.add_argument("--lookback-start", type=int, default=1440)
    parser.add_argument("--lookback-end", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parts = []
    for input_value in args.inputs:
        input_path = Path(input_value)
        extract_dir = Path(args.extract_root) / input_path.stem
        print(f"Building features from {input_path}")
        part = build_titan_features(
            input_path,
            extract_dir=extract_dir,
            table_label=args.table_label,
            lookback_start=args.lookback_start,
            lookback_end=args.lookback_end,
        )
        part["source_name"] = input_path.stem
        parts.append(part)

    features = pd.concat(parts, ignore_index=True)
    features = features.drop_duplicates("match_id")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"Wrote {len(features)} rows to {output}")


if __name__ == "__main__":
    main()
