#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.guardrails import guarded_rule_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument(
        "--rule-col",
        default="rf_away_odds_08_09_receives_05_1_active_21_50",
    )
    parser.add_argument("--output-dir", default="reports/guardrail_strong_rule_2025")
    parser.add_argument("--train-end-month", type=int, default=202509)
    parser.add_argument("--min-league-bets", type=int, default=20)
    parser.add_argument("--min-league-roi", type=float, default=0.0)
    parser.add_argument("--min-month-bets", type=int, default=50)
    parser.add_argument("--min-month-roi", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.side_dataset)
    outputs = guarded_rule_report(
        df,
        rule_col=args.rule_col,
        train_end_month=args.train_end_month,
        min_league_bets=args.min_league_bets,
        min_league_roi=args.min_league_roi,
        min_month_bets=args.min_month_bets,
        min_month_roi=args.min_month_roi,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    report = [
        "# Strong Rule Guardrail Report",
        "",
        f"Rule: `{args.rule_col}`",
        "",
        "## Overall",
        "",
        outputs["overall"].to_markdown(index=False),
        "",
        "## Monthly",
        "",
        outputs["monthly"].to_markdown(index=False),
        "",
        "## Train-Derived League Whitelist",
        "",
        outputs["whitelist"].to_markdown(index=False)
        if not outputs["whitelist"].empty
        else "No whitelist leagues.",
        "",
        "## Train-Derived Month Whitelist",
        "",
        outputs["month_whitelist"].to_markdown(index=False)
        if not outputs["month_whitelist"].empty
        else "No whitelist months.",
        "",
        "## All League Breakdown",
        "",
        outputs["league_all"].head(40).to_markdown(index=False)
        if not outputs["league_all"].empty
        else "No league rows.",
    ]
    (output_dir / "guardrail_report.md").write_text("\n".join(report), encoding="utf-8")
    print(outputs["overall"].to_string(index=False))
    print(f"Wrote guardrail outputs to {output_dir}")


if __name__ == "__main__":
    main()
