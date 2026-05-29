#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.rules import apply_rules_by_period


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--side", choices=["home", "away"], default="home")
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    features["month"] = pd.to_numeric(features["date"], errors="coerce").astype("Int64") // 100
    rules = pd.read_csv(args.rules).head(args.top)
    monthly = apply_rules_by_period(features, rules, side=args.side, period_col="month")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output, index=False)
    print(f"Wrote {len(monthly)} monthly rule rows to {output}")
    if not monthly.empty:
        summary = (
            monthly.groupby("rule")
            .agg(
                months=("period", "nunique"),
                total_bets=("bets", "sum"),
                avg_monthly_roi=("roi", "mean"),
                positive_months=("roi", lambda s: int((s > 0).sum())),
                min_monthly_roi=("roi", "min"),
                max_monthly_roi=("roi", "max"),
            )
            .reset_index()
            .sort_values(["positive_months", "avg_monthly_roi"], ascending=[False, False])
        )
        print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
