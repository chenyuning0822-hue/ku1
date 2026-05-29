#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.backtest import make_signals, summarize_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored = pd.read_csv(args.signals)
    signals = make_signals(scored, threshold=args.threshold)
    summary = summarize_backtest(signals)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote summary to {output}")


if __name__ == "__main__":
    main()

