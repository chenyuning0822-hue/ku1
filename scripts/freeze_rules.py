#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-bets", type=int, default=80)
    parser.add_argument("--min-period-bets", type=int, default=30)
    parser.add_argument("--min-period-roi", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = pd.read_csv(args.validation)
    filtered = validation[
        (validation["all_bets"] >= args.min_bets)
        & (validation["early_bets"] >= args.min_period_bets)
        & (validation["late_bets"] >= args.min_period_bets)
        & (validation["early_roi"] >= args.min_period_roi)
        & (validation["late_roi"] >= args.min_period_roi)
    ].head(args.top)

    rules = filtered.rename(columns={"all_bets": "bets", "all_roi": "roi"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(output, index=False)
    print(f"Wrote {len(rules)} frozen rules to {output}")
    if not rules.empty:
        print(rules[["rule", "bets", "roi", "early_roi", "late_roi"]].to_string(index=False))


if __name__ == "__main__":
    main()
