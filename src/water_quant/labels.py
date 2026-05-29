from __future__ import annotations

import pandas as pd

from .io import require_columns


def normalize_results(
    results: pd.DataFrame,
    *,
    match_id_col: str,
    home_score_col: str,
    away_score_col: str,
) -> pd.DataFrame:
    require_columns(results, [match_id_col, home_score_col, away_score_col], "results")
    df = results.rename(
        columns={
            match_id_col: "match_id",
            home_score_col: "home_score",
            away_score_col: "away_score",
        }
    ).copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df.dropna(subset=["match_id", "home_score", "away_score"])


def _line_profit(adjusted_margin: float, odds: float) -> float:
    if adjusted_margin > 0:
        return odds
    if adjusted_margin == 0:
        return 0.0
    return -1.0


def asian_handicap_profit(margin: float, handicap: float, odds: float) -> float:
    quarter = round(handicap * 4)
    if quarter % 2:
        lower_half_line = (quarter - 1) / 4
        upper_half_line = (quarter + 1) / 4
        return 0.5 * (
            _line_profit(margin + lower_half_line, odds)
            + _line_profit(margin + upper_half_line, odds)
        )

    adjusted = margin + handicap
    return _line_profit(adjusted, odds)


def attach_home_handicap_target(features: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(results, on="match_id", how="inner")
    if df.empty:
        raise ValueError("No feature rows matched result rows by match_id.")
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["target_home_profit"] = df.apply(
        lambda row: asian_handicap_profit(
            row["home_margin"], row["handicap_last"], row["home_odds_last"]
        ),
        axis=1,
    )
    return df
