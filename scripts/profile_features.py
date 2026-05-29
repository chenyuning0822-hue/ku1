#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.profiling import baseline_rules, profile_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", default="reports/profile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = profile_features(df)
    tables["baseline_rules"] = baseline_rules(df)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    print(f"Samples: {len(df)}")
    print("\nProfit summary")
    print(tables["profit_summary"].to_string(index=False))
    print("\nTop baseline rules")
    print(tables["baseline_rules"].head(10).to_string(index=False))
    print(f"\nWrote profile CSVs to {output_dir}")


if __name__ == "__main__":
    main()
