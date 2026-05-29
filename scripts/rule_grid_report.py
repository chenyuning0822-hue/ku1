#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.league import rule_month_league_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--side", choices=["home", "away"], default="away")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-bets", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    grid = rule_month_league_grid(
        features, args.rule, side=args.side, min_bets=args.min_bets
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=False)
    print(f"Wrote {len(grid)} month/league rows to {output}")
    if not grid.empty:
        monthly = (
            grid.groupby("month")
            .agg(
                leagues=("league", "nunique"),
                bets=("bets", "sum"),
                profit=("profit", "sum"),
                positive_leagues=("roi", lambda s: int((s > 0).sum())),
            )
            .assign(roi=lambda x: x["profit"] / x["bets"])
            .reset_index()
        )
        print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
