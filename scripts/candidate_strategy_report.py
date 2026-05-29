#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.guardrails import candidate_strategy_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument(
        "--rule-col",
        default="rf_away_odds_08_09_receives_05_1_active_21_50",
    )
    parser.add_argument("--output-dir", default="reports/candidate_strategy_2025")
    parser.add_argument("--train-end-month", type=int, default=202509)
    return parser.parse_args()


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.side_dataset)
    outputs = candidate_strategy_report(
        df,
        rule_col=args.rule_col,
        train_end_month=args.train_end_month,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    overall = outputs["overall"].copy()
    monthly = outputs["monthly"].copy()
    sensitivity = outputs["league_threshold_sensitivity"].copy()
    best_sensitivity = sensitivity[sensitivity["bets"] >= 50].head(10) if not sensitivity.empty else sensitivity

    report = [
        "# Candidate Strategy Report",
        "",
        f"Rule: `{args.rule_col}`",
        f"Train period: months <= `{args.train_end_month}`. Test period: later months.",
        "",
        "## Decision Snapshot",
        "",
    ]
    if not overall.empty:
        test_row = overall[overall["scenario"] == "test_period"].iloc[0]
        report.extend(
            [
                f"- Test bets: {int(test_row['bets'])}",
                f"- Test ROI: {_format_pct(float(test_row['roi']))}",
                f"- Test profit: {float(test_row['profit']):.2f}",
                f"- Test max drawdown: {float(test_row['max_drawdown']):.2f}",
                "",
            ]
        )

    report.extend(
        [
            "## Overall",
            "",
            overall.to_markdown(index=False) if not overall.empty else "No rows.",
            "",
            "## Monthly",
            "",
            monthly.to_markdown(index=False) if not monthly.empty else "No rows.",
            "",
            "## League Guardrail Sensitivity",
            "",
            best_sensitivity.to_markdown(index=False)
            if not best_sensitivity.empty
            else "No threshold setting kept at least 50 test bets.",
            "",
            "## Largest Leagues",
            "",
            outputs["league"].head(30).to_markdown(index=False)
            if not outputs["league"].empty
            else "No league rows.",
        ]
    )
    (output_dir / "candidate_strategy_report.md").write_text("\n".join(report), encoding="utf-8")

    print(overall.to_string(index=False))
    print(f"Wrote candidate strategy outputs to {output_dir}")


if __name__ == "__main__":
    main()
