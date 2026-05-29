#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_side_baseline import build_pipeline
from water_quant.ml_baseline import (
    add_month,
    available_features,
    scan_thresholds,
    summarize_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument("--output-dir", default="reports/ml_walk_forward_2025")
    parser.add_argument("--start-test-month", type=int, default=202504)
    parser.add_argument("--min-bets", type=int, default=300)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = add_month(pd.read_csv(args.side_dataset))
    numeric, categorical = available_features(data)
    months = sorted(int(m) for m in data["month"].dropna().unique())
    rows = []
    scans = []

    for test_month in months:
        if test_month < args.start_test_month:
            continue
        train = data[data["month"] < test_month].copy()
        test = data[data["month"] == test_month].copy()
        if train.empty or test.empty:
            continue

        model = build_pipeline(
            numeric,
            categorical,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
        )
        model.fit(train[numeric + categorical], train["target_profit"])
        test["prediction"] = model.predict(test[numeric + categorical])

        threshold_scan = scan_thresholds(test, min_bets=args.min_bets)
        threshold_scan["test_month"] = test_month
        scans.append(threshold_scan)

        eligible = threshold_scan[threshold_scan["eligible"]]
        best = eligible.iloc[0] if not eligible.empty else threshold_scan.iloc[0]
        summary = summarize_predictions(test, threshold=float(best["threshold"]))
        summary["test_month"] = test_month
        summary["train_rows"] = int(len(train))
        summary["test_rows"] = int(len(test))
        rows.append(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    scan_result = pd.concat(scans, ignore_index=True) if scans else pd.DataFrame()
    result.to_csv(output_dir / "walk_forward_summary.csv", index=False)
    scan_result.to_csv(output_dir / "walk_forward_threshold_scans.csv", index=False)

    aggregate = pd.DataFrame(
        [
            {
                "months": int(len(result)),
                "bets": int(result["bets"].sum()) if not result.empty else 0,
                "profit": float(result["profit"].sum()) if not result.empty else 0.0,
                "roi": float(result["profit"].sum() / result["bets"].sum())
                if not result.empty and result["bets"].sum()
                else 0.0,
                "positive_months": int((result["roi"] > 0).sum()) if not result.empty else 0,
            }
        ]
    )
    aggregate.to_csv(output_dir / "walk_forward_aggregate.csv", index=False)

    report = [
        "# Walk-Forward Sklearn Side Baseline",
        "",
        "Each test month is predicted using only prior months.",
        "",
        "## Aggregate",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Monthly Summary",
        "",
        result.to_markdown(index=False) if not result.empty else "No rows.",
    ]
    (output_dir / "walk_forward_report.md").write_text("\n".join(report), encoding="utf-8")

    print(aggregate.to_string(index=False))
    print(result.to_string(index=False) if not result.empty else "No rows.")


if __name__ == "__main__":
    main()
