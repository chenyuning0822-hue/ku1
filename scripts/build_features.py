#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from water_quant.features import build_match_features, normalize_odds_data
from water_quant.io import read_table
from water_quant.labels import attach_home_handicap_target, normalize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--match-id-col", default="match_id")
    parser.add_argument("--snapshot-time-col", default="snapshot_time")
    parser.add_argument("--kickoff-time-col", default="kickoff_time")
    parser.add_argument("--handicap-col", default="handicap")
    parser.add_argument("--home-odds-col", default="home_odds")
    parser.add_argument("--away-odds-col", default="away_odds")
    parser.add_argument("--home-score-col", default="home_score")
    parser.add_argument("--away-score-col", default="away_score")
    parser.add_argument("--lookback-start", type=int, default=60)
    parser.add_argument("--lookback-end", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    odds = normalize_odds_data(
        read_table(args.odds),
        match_id_col=args.match_id_col,
        snapshot_time_col=args.snapshot_time_col,
        kickoff_time_col=args.kickoff_time_col,
        handicap_col=args.handicap_col,
        home_odds_col=args.home_odds_col,
        away_odds_col=args.away_odds_col,
    )
    results = normalize_results(
        read_table(args.results),
        match_id_col=args.match_id_col,
        home_score_col=args.home_score_col,
        away_score_col=args.away_score_col,
    )
    features = build_match_features(
        odds, lookback_start=args.lookback_start, lookback_end=args.lookback_end
    )
    labeled = attach_home_handicap_target(features, results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output, index=False)
    print(f"Wrote {len(labeled)} rows to {output}")


if __name__ == "__main__":
    main()

