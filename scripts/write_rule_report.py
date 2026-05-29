#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="reports/rule_report.md")
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = pd.read_csv(args.rules)
    validation = pd.read_csv(args.validation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Rule Search Report",
        "",
        "This report uses rule segments only. No machine-learning training is involved.",
        "",
        "## Top In-Sample Segments",
        "",
        rules.head(args.top)[
            ["rule", "bets", "roi", "win_rate", "push_rate", "avg_home_odds", "avg_handicap"]
        ].to_markdown(index=False),
        "",
        "## Top Period-Validated Segments",
        "",
        "Rules below rank by the worse ROI between the first and second half of the month.",
        "",
        validation.head(args.top)[
            ["rule", "all_bets", "all_roi", "early_bets", "early_roi", "late_bets", "late_roi"]
        ].to_markdown(index=False),
        "",
        "## Reading Notes",
        "",
        "- `line_move=same`: opening/first line and latest line are the same bucket value.",
        "- `home_stronger`: handicap moved against the home side, meaning the home side gives more or receives less.",
        "- `home_weaker`: handicap moved in favor of the home side, meaning the home side gives less or receives more.",
        "- Positive ROI here is exploratory. It must be checked on later months before use.",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
