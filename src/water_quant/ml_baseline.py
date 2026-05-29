from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NUMERIC_FEATURES = [
    "is_home_side",
    "side_handicap_first",
    "side_handicap_last",
    "side_handicap_change",
    "side_odds_first",
    "side_odds_last",
    "side_odds_change",
    "first_minute_to_start",
    "last_minute_to_start",
    "row_count",
    "handicap_std",
    "handicap_range",
    "handicap_abs_change_sum",
    "home_odds_std",
    "home_odds_range",
    "home_odds_abs_change_sum",
    "away_odds_std",
    "away_odds_range",
    "away_odds_abs_change_sum",
]

CATEGORICAL_FEATURES = [
    "side",
    "league",
    "line_move",
    "odds_move",
    "side_odds_bucket",
    "side_handicap_bucket",
    "snapshot_bucket",
    "activity_bucket",
]


@dataclass(frozen=True)
class SplitConfig:
    train_end_month: int = 202509
    valid_start_month: int = 202510
    valid_end_month: int = 202511
    test_month: int = 202512


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = pd.to_numeric(out["date"], errors="coerce").astype("Int64") // 100
    return out


def time_split(df: pd.DataFrame, config: SplitConfig = SplitConfig()) -> dict[str, pd.DataFrame]:
    data = add_month(df)
    train = data[data["month"] <= config.train_end_month].copy()
    valid = data[
        (data["month"] >= config.valid_start_month)
        & (data["month"] <= config.valid_end_month)
    ].copy()
    test = data[data["month"] == config.test_month].copy()
    return {"train": train, "valid": valid, "test": test}


def available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    return numeric, categorical


def summarize_predictions(
    df: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    threshold: float,
) -> dict[str, float | int]:
    bets = df[df[prediction_col] >= threshold].copy()
    if bets.empty:
        return {
            "threshold": threshold,
            "bets": 0,
            "profit": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "push_rate": 0.0,
            "lose_rate": 0.0,
            "avg_prediction": 0.0,
        }
    profit = bets["target_profit"]
    return {
        "threshold": threshold,
        "bets": int(len(bets)),
        "profit": float(profit.sum()),
        "roi": float(profit.mean()),
        "win_rate": float((profit > 0).mean()),
        "push_rate": float((profit == 0).mean()),
        "lose_rate": float((profit < 0).mean()),
        "avg_prediction": float(bets[prediction_col].mean()),
    }


def scan_thresholds(
    df: pd.DataFrame,
    *,
    thresholds: list[float] | None = None,
    min_bets: int = 100,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    rows = [summarize_predictions(df, threshold=threshold) for threshold in thresholds]
    result = pd.DataFrame(rows)
    result["eligible"] = result["bets"] >= min_bets
    return result.sort_values(["eligible", "roi", "bets"], ascending=[False, False, False])


def month_report(df: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    data = add_month(df)
    rows = []
    for (month, side), group in data.groupby(["month", "side"], dropna=False):
        summary = summarize_predictions(group, threshold=threshold)
        summary["month"] = int(month)
        summary["side"] = side
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["month", "side"])
