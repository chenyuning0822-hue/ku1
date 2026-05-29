#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from water_quant.ml_baseline import (
    SplitConfig,
    month_report,
    scan_thresholds,
    time_split,
)
from water_quant.table_model import SmoothedTableModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument("--output-dir", default="reports/table_baseline_2025")
    parser.add_argument("--model-output", default="models/side_profit_table_model.pkl")
    parser.add_argument("--min-bets", type=int, default=300)
    parser.add_argument("--group-min-count", type=int, default=50)
    parser.add_argument("--smoothing", type=float, default=200.0)
    return parser.parse_args()


def write_report(
    output: Path,
    *,
    split_sizes: dict[str, int],
    validation_scan: pd.DataFrame,
    test_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    model: SmoothedTableModel,
) -> None:
    lines = [
        "# Side Profit Table Baseline",
        "",
        "This is a no-sklearn baseline using smoothed historical profit tables.",
        "",
        "## Split Sizes",
        "",
        pd.DataFrame([split_sizes]).to_markdown(index=False),
        "",
        "## Model Tables",
        "",
        pd.DataFrame(
            [
                {
                    "columns": ", ".join(rule.columns),
                    "rows": len(rule.table),
                    "smoothing": rule.smoothing,
                    "min_count": rule.min_count,
                }
                for rule in model.rules
            ]
        ).to_markdown(index=False),
        "",
        "## Validation Threshold Scan",
        "",
        validation_scan.to_markdown(index=False),
        "",
        "## Test Summary",
        "",
        test_summary.to_markdown(index=False),
        "",
        "## Monthly Test Breakdown",
        "",
        monthly.to_markdown(index=False) if not monthly.empty else "No rows.",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.side_dataset)
    splits = time_split(data, SplitConfig())

    model = SmoothedTableModel(
        smoothing=args.smoothing,
        min_count=args.group_min_count,
    )
    model.fit(splits["train"])
    for split_df in splits.values():
        split_df["prediction"] = model.predict(split_df)

    validation_scan = scan_thresholds(splits["valid"], min_bets=args.min_bets)
    eligible = validation_scan[validation_scan["eligible"]]
    best_threshold = float(
        eligible.iloc[0]["threshold"] if not eligible.empty else validation_scan.iloc[0]["threshold"]
    )
    test_summary = pd.DataFrame(
        [
            {
                "chosen_threshold": best_threshold,
                **scan_thresholds(splits["test"], thresholds=[best_threshold], min_bets=0).iloc[0].to_dict(),
            }
        ]
    )
    monthly = month_report(splits["test"], threshold=best_threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_scan.to_csv(output_dir / "validation_threshold_scan.csv", index=False)
    test_summary.to_csv(output_dir / "test_summary.csv", index=False)
    monthly.to_csv(output_dir / "monthly_test_breakdown.csv", index=False)
    for name, table in model.export_tables().items():
        table.to_csv(output_dir / f"table_{name}.csv", index=False)

    write_report(
        output_dir / "side_profit_table_baseline_report.md",
        split_sizes={name: len(split) for name, split in splits.items()},
        validation_scan=validation_scan,
        test_summary=test_summary,
        monthly=monthly,
        model=model,
    )

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as fh:
        pickle.dump(model, fh)

    print(f"Wrote report to {output_dir / 'side_profit_table_baseline_report.md'}")
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()
