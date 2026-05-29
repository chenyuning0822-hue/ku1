#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from water_quant.ml_baseline import (
    SplitConfig,
    available_features,
    month_report,
    scan_thresholds,
    time_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-dataset", required=True)
    parser.add_argument("--output-dir", default="reports/ml_baseline")
    parser.add_argument("--model-output", default="models/side_profit_baseline.pkl")
    parser.add_argument("--min-bets", type=int, default=300)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    return parser.parse_args()


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    n_estimators: int = 300,
    min_samples_leaf: int = 50,
):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing scikit-learn. Install dependencies with: "
            "python3 -m pip install -r requirements-dev.txt"
        ) from exc

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=0.8,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def write_report(
    output: Path,
    *,
    split_sizes: dict[str, int],
    features: tuple[list[str], list[str]],
    validation_scan: pd.DataFrame,
    test_summary: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    numeric, categorical = features
    lines = [
        "# Side Profit ML Baseline",
        "",
        "Target: `target_profit` regression on side-level samples.",
        "",
        "## Split Sizes",
        "",
        pd.DataFrame([split_sizes]).to_markdown(index=False),
        "",
        "## Features",
        "",
        f"Numeric: {', '.join(numeric)}",
        "",
        f"Categorical: {', '.join(categorical)}",
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
    numeric, categorical = available_features(data)
    model = build_pipeline(
        numeric,
        categorical,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
    )

    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    X_train = train[numeric + categorical]
    y_train = train["target_profit"]
    model.fit(X_train, y_train)

    for split_name, split_df in splits.items():
        split_df["prediction"] = model.predict(split_df[numeric + categorical])

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
        output_dir / "side_profit_baseline_report.md",
        split_sizes={name: len(split) for name, split in splits.items()},
        features=(numeric, categorical),
        validation_scan=validation_scan,
        test_summary=test_summary,
        monthly=monthly,
    )

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as fh:
        pickle.dump({"model": model, "numeric": numeric, "categorical": categorical}, fh)

    print(f"Wrote report to {output_dir / 'side_profit_baseline_report.md'}")
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()
