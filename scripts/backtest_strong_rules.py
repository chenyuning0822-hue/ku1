#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.ml_baseline import add_month
from water_quant.rule_features import add_strong_rule_features, strong_rule_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument("--output-dir", default="reports/strong_rule_backtest_2025")
    return parser.parse_args()


def summarize(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, float | int | str]:
    bets = df[mask].copy()
    profit = bets["target_profit"] if not bets.empty else pd.Series(dtype=float)
    return {
        "rule": name,
        "bets": int(len(bets)),
        "profit": float(profit.sum()) if len(bets) else 0.0,
        "roi": float(profit.mean()) if len(bets) else 0.0,
        "win_rate": float((profit > 0).mean()) if len(bets) else 0.0,
        "push_rate": float((profit == 0).mean()) if len(bets) else 0.0,
        "lose_rate": float((profit < 0).mean()) if len(bets) else 0.0,
    }


def main() -> None:
    args = parse_args()
    df = add_month(add_strong_rule_features(pd.read_csv(args.side_dataset)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    monthly_rows = []
    for rule in strong_rule_columns():
        mask = df[rule] == 1
        rows.append(summarize(df, mask, rule))
        for month, group in df.groupby("month"):
            monthly_rows.append(summarize(group, group[rule] == 1, rule) | {"month": int(month)})

    overall = pd.DataFrame(rows).sort_values(["roi", "bets"], ascending=[False, False])
    monthly = pd.DataFrame(monthly_rows).sort_values(["rule", "month"])
    overall.to_csv(output_dir / "strong_rule_overall.csv", index=False)
    monthly.to_csv(output_dir / "strong_rule_monthly.csv", index=False)

    report = [
        "# Strong Rule Backtest",
        "",
        "Pure rule backtest. No machine-learning model is used.",
        "",
        "## Overall",
        "",
        overall.to_markdown(index=False),
        "",
        "## Monthly",
        "",
        monthly.to_markdown(index=False),
    ]
    (output_dir / "strong_rule_backtest_report.md").write_text("\n".join(report), encoding="utf-8")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
