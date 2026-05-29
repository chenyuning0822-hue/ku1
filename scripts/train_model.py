#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from water_quant.modeling import train_profit_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--signals-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features)
    model, cols, metrics = train_profit_model(df)
    df["predicted_home_profit"] = model.predict(df[cols])

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "feature_columns": cols, "metrics": metrics}
    try:
        import joblib

        joblib.dump(artifact, model_output)
    except ModuleNotFoundError:
        with model_output.open("wb") as fh:
            pickle.dump(artifact, fh)

    signals_output = Path(args.signals_output)
    signals_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(signals_output, index=False)
    print(f"Metrics: {metrics}")
    print(f"Wrote model to {model_output}")
    print(f"Wrote scored rows to {signals_output}")


if __name__ == "__main__":
    main()
