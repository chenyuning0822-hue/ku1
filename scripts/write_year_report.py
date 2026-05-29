#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt_pct(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", default="reports/profile_2025_all")
    parser.add_argument("--frozen-apply", default="reports/frozen_rules_2025_exnov_apply.csv")
    parser.add_argument("--monthly", default="reports/frozen_rules_2025_monthly.csv")
    parser.add_argument("--large-rules", default="reports/rule_search_2025_all_min1000.csv")
    parser.add_argument("--large-validation", default="reports/rule_validation_2025_all_min1000.csv")
    parser.add_argument("--output", default="reports/year_2025_research_report.md")
    return parser.parse_args()


def table(df: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    return df.head(n)[columns].to_markdown(index=False)


def main() -> None:
    args = parse_args()
    profile_dir = Path(args.profile_dir)
    summary = pd.read_csv(profile_dir / "profit_summary.csv")
    frozen = pd.read_csv(args.frozen_apply)
    monthly = pd.read_csv(args.monthly)
    large = pd.read_csv(args.large_rules)
    large_validation = pd.read_csv(args.large_validation)

    monthly_summary = (
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

    lines = [
        "# 2025 Research Report",
        "",
        "No machine-learning training was used in this report.",
        "",
        "## Full-Year Baseline",
        "",
        table(
            summary,
            [
                "samples",
                "mean_profit",
                "win_rate",
                "push_rate",
                "lose_rate",
                "min_profit",
                "max_profit",
            ],
            1,
        ),
        "",
        "## Frozen November Rules Tested Outside November",
        "",
        table(
            frozen,
            [
                "rule",
                "source_roi",
                "test_bets",
                "test_roi",
                "test_win_rate",
                "test_push_rate",
            ],
            12,
        ),
        "",
        "## Frozen Rule Monthly Stability",
        "",
        table(
            monthly_summary,
            [
                "rule",
                "months",
                "total_bets",
                "avg_monthly_roi",
                "positive_months",
                "min_monthly_roi",
                "max_monthly_roi",
            ],
            12,
        ),
        "",
        "## Full-Year Large-Sample Rule Search",
        "",
        table(
            large,
            ["rule", "bets", "roi", "win_rate", "push_rate", "avg_home_odds", "avg_handicap"],
            12,
        ),
        "",
        "## Large-Sample Period Validation",
        "",
        table(
            large_validation,
            ["rule", "all_bets", "all_roi", "early_bets", "early_roi", "late_bets", "late_roi"],
            12,
        ),
        "",
        "## Notes",
        "",
        "- The November-only high ROI rules mostly did not generalize outside November.",
        "- The best large-sample full-year segments are much smaller, around low single-digit ROI.",
        "- Before model training, the next useful step is to test away-side symmetry and league filters.",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
