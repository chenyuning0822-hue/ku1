#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.live_candidates import (
    DEFAULT_RULE_COL,
    latest_date,
    select_live_candidates,
    summarize_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument("--output", default="reports/live_candidates/latest_candidates.csv")
    parser.add_argument("--date", type=int, default=None, help="YYYYMMDD. Defaults to latest date in data.")
    parser.add_argument("--rule-col", default=DEFAULT_RULE_COL)
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Keep result/profit columns for historical review. Do not use for pre-match tracking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.side_dataset)
    target_date = args.date if args.date is not None else latest_date(df)
    candidates = select_live_candidates(
        df,
        rule_col=args.rule_col,
        target_date=target_date,
        include_results=args.include_results,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output, index=False)

    summary = summarize_candidates(candidates)
    print(f"Date: {target_date}")
    print(f"Rule: {args.rule_col}")
    print(f"Candidates: {summary['candidates']}")
    if "profit" in summary:
        print(f"Historical profit: {summary['profit']:.2f}")
        print(f"Historical ROI: {summary['roi'] * 100:.2f}%")
    print(f"Wrote candidates to {output}")
    if not candidates.empty:
        display_cols = [
            col
            for col in [
                "start_time_text",
                "league",
                "home_team",
                "away_team",
                "side",
                "side_handicap_last",
                "side_odds_last",
                "activity_bucket",
            ]
            if col in candidates.columns
        ]
        print(candidates[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
