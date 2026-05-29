#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-profile", default="reports/profile_2025_all/profit_summary.csv")
    parser.add_argument("--away-profile", default="reports/profile_2025_all_away/profit_summary.csv")
    parser.add_argument("--home-rules", default="reports/rule_validation_2025_all_min1000.csv")
    parser.add_argument("--away-rules", default="reports/rule_validation_2025_all_away_min1000.csv")
    parser.add_argument("--output", default="reports/side_symmetry_2025_report.md")
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def table(df: pd.DataFrame, columns: list[str], n: int) -> str:
    return df.head(n)[columns].to_markdown(index=False)


def main() -> None:
    args = parse_args()
    home_profile = pd.read_csv(args.home_profile)
    away_profile = pd.read_csv(args.away_profile)
    home_rules = pd.read_csv(args.home_rules)
    away_rules = pd.read_csv(args.away_rules)

    lines = [
        "# 2025 Side Symmetry Report",
        "",
        "No machine-learning training was used in this report.",
        "",
        "## Baseline",
        "",
        "| side | samples | mean_profit | win_rate | push_rate | lose_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for side, row in [("home", home_profile.iloc[0]), ("away", away_profile.iloc[0])]:
        lines.append(
            f"| {side} | {int(row['samples'])} | {row['mean_profit']:.4f} | "
            f"{row['win_rate']:.4f} | {row['push_rate']:.4f} | {row['lose_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Home Large-Sample Period Validation",
            "",
            table(
                home_rules,
                ["rule", "all_bets", "all_roi", "early_roi", "late_roi"],
                args.top,
            ),
            "",
            "## Away Large-Sample Period Validation",
            "",
            table(
                away_rules,
                ["rule", "all_bets", "all_roi", "early_roi", "late_roi"],
                args.top,
            ),
            "",
            "## Notes",
            "",
            "- Away-side large-sample rules are stronger than the home-side rules in this 2025 dataset.",
            "- Use side-neutral `side_*` rule names for future ML features so home and away samples can be stacked.",
            "- The next step is league filtering on these side-neutral candidate rules.",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
