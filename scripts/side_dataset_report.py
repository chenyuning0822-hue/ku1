#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument("--output", default="reports/side_dataset_report.md")
    return parser.parse_args()


def as_table(df: pd.DataFrame, n: int = 20) -> str:
    return df.head(n).to_markdown(index=False)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.side_dataset)
    df["month"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64") // 100

    pair_counts = df.groupby("match_id")["side"].nunique().value_counts().reset_index()
    pair_counts.columns = ["unique_sides_per_match", "matches"]

    side_summary = (
        df.groupby("side")
        .agg(
            samples=("target_profit", "size"),
            mean_profit=("target_profit", "mean"),
            win_rate=("target_profit", lambda s: (s > 0).mean()),
            push_rate=("target_profit", lambda s: (s == 0).mean()),
            lose_rate=("target_profit", lambda s: (s < 0).mean()),
        )
        .reset_index()
    )

    month_summary = (
        df.groupby(["month", "side"])
        .agg(
            samples=("target_profit", "size"),
            mean_profit=("target_profit", "mean"),
            win_rate=("target_profit", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )

    category_summary = pd.DataFrame(
        [
            {"column": col, "unique_values": int(df[col].nunique(dropna=False))}
            for col in [
                "league",
                "side",
                "line_move",
                "odds_move",
                "side_odds_bucket",
                "side_handicap_bucket",
                "snapshot_bucket",
                "activity_bucket",
            ]
            if col in df.columns
        ]
    )

    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Side Dataset Report",
        "",
        "This dataset is the machine-learning-ready side-level table: one match becomes one home-side row and one away-side row.",
        "",
        "## Pair Completeness",
        "",
        as_table(pair_counts),
        "",
        "## Side Summary",
        "",
        as_table(side_summary),
        "",
        "## Month Summary",
        "",
        as_table(month_summary, 30),
        "",
        "## Category Cardinality",
        "",
        as_table(category_summary),
        "",
        "## Missing Values",
        "",
        as_table(missing[missing["missing_rate"] > 0], 30)
        if (missing["missing_rate"] > 0).any()
        else "No missing values.",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")
    print(side_summary.to_string(index=False))


if __name__ == "__main__":
    main()
