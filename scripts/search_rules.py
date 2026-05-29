#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.rules import search_rule_segments, validate_rules_by_period


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default="reports/rule_search.csv")
    parser.add_argument("--min-bets", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--validation-output", default=None)
    parser.add_argument("--top-n-validation", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features)
    rules = search_rule_segments(df, min_bets=args.min_bets, max_depth=args.max_depth)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(output, index=False)
    print(f"Wrote {len(rules)} candidate rule segments to {output}")
    if not rules.empty:
        print(rules.head(20).to_string(index=False))
    if args.validation_output:
        validation = validate_rules_by_period(
            df, rules, top_n=args.top_n_validation
        )
        validation_output = Path(args.validation_output)
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation.to_csv(validation_output, index=False)
        print(f"\nWrote period validation to {validation_output}")
        if not validation.empty:
            print(validation.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
