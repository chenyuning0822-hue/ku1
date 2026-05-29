#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from water_quant.titan import build_titan_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Titan sqlite file, sqlite directory, or zip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract-dir", default="data/raw/titan_extracted")
    parser.add_argument("--table-label", default="亚盘")
    parser.add_argument("--lookback-start", type=int, default=1440)
    parser.add_argument("--lookback-end", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_titan_features(
        args.input,
        extract_dir=args.extract_dir,
        table_label=args.table_label,
        lookback_start=args.lookback_start,
        lookback_end=args.lookback_end,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"Wrote {len(features)} rows to {output}")


if __name__ == "__main__":
    main()
