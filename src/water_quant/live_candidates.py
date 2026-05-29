from __future__ import annotations

import pandas as pd

from .rule_features import add_strong_rule_features


DEFAULT_RULE_COL = "rf_away_odds_08_09_receives_05_1_active_21_50"

CANDIDATE_COLUMNS = [
    "date",
    "match_id",
    "league",
    "home_team",
    "away_team",
    "start_time_text",
    "side",
    "side_handicap_last",
    "side_odds_last",
    "side_handicap_bucket",
    "side_odds_bucket",
    "activity_bucket",
    "snapshot_bucket",
    "line_move",
    "odds_move",
    "last_minute_to_start",
    "row_count",
    "target_profit",
    "final_score",
]


def latest_date(df: pd.DataFrame) -> int:
    dates = pd.to_numeric(df["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("No usable date values found.")
    return int(dates.max())


def select_live_candidates(
    df: pd.DataFrame,
    *,
    rule_col: str = DEFAULT_RULE_COL,
    target_date: int | None = None,
    include_results: bool = False,
) -> pd.DataFrame:
    data = add_strong_rule_features(df)
    if rule_col not in data.columns:
        raise ValueError(f"Missing rule column: {rule_col}")

    if target_date is None:
        target_date = latest_date(data)

    date_values = pd.to_numeric(data["date"], errors="coerce").astype("Int64")
    candidates = data[(date_values == target_date) & (data[rule_col] == 1)].copy()
    if not include_results:
        candidates = candidates.drop(
            columns=[col for col in ["target_profit", "final_score"] if col in candidates.columns]
        )

    keep = [col for col in CANDIDATE_COLUMNS if col in candidates.columns]
    candidates = candidates[keep]
    sort_cols = [col for col in ["start_time_text", "league", "match_id"] if col in candidates.columns]
    return candidates.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def summarize_candidates(candidates: pd.DataFrame) -> dict[str, int | float]:
    summary: dict[str, int | float] = {"candidates": int(len(candidates))}
    if "target_profit" in candidates.columns:
        profit = candidates["target_profit"]
        summary["profit"] = float(profit.sum()) if len(profit) else 0.0
        summary["roi"] = float(profit.mean()) if len(profit) else 0.0
    return summary
