from __future__ import annotations

import numpy as np
import pandas as pd

from .io import require_columns


def normalize_odds_data(
    odds: pd.DataFrame,
    *,
    match_id_col: str,
    snapshot_time_col: str,
    kickoff_time_col: str,
    handicap_col: str,
    home_odds_col: str,
    away_odds_col: str,
) -> pd.DataFrame:
    require_columns(
        odds,
        [
            match_id_col,
            snapshot_time_col,
            kickoff_time_col,
            handicap_col,
            home_odds_col,
            away_odds_col,
        ],
        "odds",
    )
    df = odds.rename(
        columns={
            match_id_col: "match_id",
            snapshot_time_col: "snapshot_time",
            kickoff_time_col: "kickoff_time",
            handicap_col: "handicap",
            home_odds_col: "home_odds",
            away_odds_col: "away_odds",
        }
    ).copy()
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    df["minute_to_start"] = (
        df["kickoff_time"] - df["snapshot_time"]
    ).dt.total_seconds() / 60
    for col in ["handicap", "home_odds", "away_odds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=["match_id", "minute_to_start", "handicap", "home_odds", "away_odds"]
    )
    return df.sort_values(["match_id", "minute_to_start"], ascending=[True, False])


def _series_features(values: pd.Series, prefix: str) -> dict[str, float]:
    values = values.astype(float).dropna()
    if values.empty:
        return {}
    diffs = values.diff().dropna()
    return {
        f"{prefix}_first": float(values.iloc[0]),
        f"{prefix}_last": float(values.iloc[-1]),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_range": float(values.max() - values.min()),
        f"{prefix}_change": float(values.iloc[-1] - values.iloc[0]),
        f"{prefix}_abs_change_sum": float(diffs.abs().sum()) if not diffs.empty else 0.0,
        f"{prefix}_up_ticks": float((diffs > 0).sum()) if not diffs.empty else 0.0,
        f"{prefix}_down_ticks": float((diffs < 0).sum()) if not diffs.empty else 0.0,
    }


def _window_change(group: pd.DataFrame, col: str, start_minute: int, end_minute: int) -> float:
    window = group[
        (group["minute_to_start"] <= start_minute)
        & (group["minute_to_start"] >= end_minute)
    ].sort_values("minute_to_start", ascending=False)
    if len(window) < 2:
        return np.nan
    return float(window[col].iloc[-1] - window[col].iloc[0])


def _window_features(
    group: pd.DataFrame,
    *,
    start_minute: int,
    end_minute: int,
    label: str,
) -> dict[str, float]:
    window = group[
        (group["minute_to_start"] <= start_minute)
        & (group["minute_to_start"] > end_minute)
    ].sort_values("minute_to_start", ascending=False)
    if window.empty:
        return {
            f"{label}_row_count": 0.0,
        }

    row: dict[str, float] = {
        f"{label}_row_count": float(len(window)),
        f"{label}_duration": float(
            window["minute_to_start"].iloc[0] - window["minute_to_start"].iloc[-1]
        ),
    }
    for col in ["home_odds", "away_odds", "handicap"]:
        values = window[col].astype(float).dropna()
        if values.empty:
            continue
        diffs = values.diff().dropna()
        row[f"{label}_{col}_first"] = float(values.iloc[0])
        row[f"{label}_{col}_last"] = float(values.iloc[-1])
        row[f"{label}_{col}_change"] = float(values.iloc[-1] - values.iloc[0])
        row[f"{label}_{col}_mean"] = float(values.mean())
        row[f"{label}_{col}_std"] = float(values.std(ddof=0))
        row[f"{label}_{col}_range"] = float(values.max() - values.min())
        row[f"{label}_{col}_up_ticks"] = float((diffs > 0).sum()) if not diffs.empty else 0.0
        row[f"{label}_{col}_down_ticks"] = float((diffs < 0).sum()) if not diffs.empty else 0.0
        row[f"{label}_{col}_abs_change_sum"] = (
            float(diffs.abs().sum()) if not diffs.empty else 0.0
        )
        row[f"{label}_{col}_last_step"] = float(diffs.iloc[-1]) if not diffs.empty else 0.0
    return row


def build_match_features(
    odds: pd.DataFrame,
    *,
    lookback_start: int = 60,
    lookback_end: int = 5,
) -> pd.DataFrame:
    sample = odds[
        (odds["minute_to_start"] <= lookback_start)
        & (odds["minute_to_start"] >= lookback_end)
    ].copy()
    if sample.empty:
        raise ValueError(
            f"No odds rows inside lookback window {lookback_start} to {lookback_end} minutes."
        )

    rows: list[dict[str, float | str]] = []
    for match_id, group in sample.groupby("match_id", sort=False):
        group = group.sort_values("minute_to_start", ascending=False)
        row: dict[str, float | str] = {"match_id": match_id}
        row.update(_series_features(group["home_odds"], "home_odds"))
        row.update(_series_features(group["away_odds"], "away_odds"))
        row.update(_series_features(group["handicap"], "handicap"))
        row["home_away_odds_diff_last"] = (
            group["home_odds"].iloc[-1] - group["away_odds"].iloc[-1]
        )
        row["home_away_odds_sum_last"] = (
            group["home_odds"].iloc[-1] + group["away_odds"].iloc[-1]
        )
        row["first_minute_to_start"] = float(group["minute_to_start"].iloc[0])
        row["last_minute_to_start"] = float(group["minute_to_start"].iloc[-1])
        row["row_count"] = float(len(group))
        for minutes in [5, 10, 30, 60]:
            if minutes <= lookback_start:
                row[f"home_odds_change_{minutes}m"] = _window_change(
                    group, "home_odds", minutes, lookback_end
                )
                row[f"away_odds_change_{minutes}m"] = _window_change(
                    group, "away_odds", minutes, lookback_end
                )
        for start, end, label in [
            (5, 0, "w_0_5m"),
            (10, 5, "w_5_10m"),
            (30, 10, "w_10_30m"),
            (60, 30, "w_30_60m"),
            (180, 60, "w_1_3h"),
            (720, 180, "w_3_12h"),
            (1440, 720, "w_12_24h"),
        ]:
            if start <= lookback_start and end >= lookback_end:
                row.update(
                    _window_features(
                        group, start_minute=start, end_minute=end, label=label
                    )
                )
        rows.append(row)

    return pd.DataFrame(rows)
