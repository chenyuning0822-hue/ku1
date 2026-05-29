#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_side_baseline import build_pipeline, write_report
from water_quant.ml_baseline import SplitConfig, core_features, month_report, scan_thresholds, time_split
from water_quant.rule_features import add_strong_rule_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument("--output-dir", default="reports/rule_feature_baseline_2025")
    parser.add_argument("--min-bets", type=int, default=300)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = add_strong_rule_features(pd.read_csv(args.side_dataset))
    splits = time_split(data, SplitConfig())
    numeric, categorical = core_features(data)
    model = build_pipeline(
        numeric,
        categorical,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
    )
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]
    model.fit(train[numeric + categorical], train["target_profit"])
    for split in splits.values():
        split["prediction"] = model.predict(split[numeric + categorical])

    validation_scan = scan_thresholds(valid, min_bets=args.min_bets)
    eligible = validation_scan[validation_scan["eligible"]]
    best_threshold = float(
        eligible.iloc[0]["threshold"] if not eligible.empty else validation_scan.iloc[0]["threshold"]
    )
    test_summary = pd.DataFrame(
        [
            {
                "chosen_threshold": best_threshold,
                **scan_thresholds(test, thresholds=[best_threshold], min_bets=0).iloc[0].to_dict(),
            }
        ]
    )
    monthly = month_report(test, threshold=best_threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_scan.to_csv(output_dir / "validation_threshold_scan.csv", index=False)
    test_summary.to_csv(output_dir / "test_summary.csv", index=False)
    monthly.to_csv(output_dir / "monthly_test_breakdown.csv", index=False)
    write_report(
        output_dir / "rule_feature_baseline_report.md",
        split_sizes={name: len(split) for name, split in splits.items()},
        features=(numeric, categorical),
        validation_scan=validation_scan,
        test_summary=test_summary,
        monthly=monthly,
    )
    print(f"Wrote report to {output_dir / 'rule_feature_baseline_report.md'}")
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()
