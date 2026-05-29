from __future__ import annotations

import pandas as pd


def add_strong_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rf_side_weaker_0_5m_active_21_50"] = (
        (out["line_move"] == "side_weaker")
        & (out["snapshot_bucket"] == "0-5m")
        & (out["activity_bucket"] == "21-50")
    ).astype(int)
    out["rf_away_receives_05_1_active_21_50"] = (
        (out["side"] == "away")
        & (out["side_handicap_bucket"] == "side_receives_0.5_to_1")
        & (out["activity_bucket"] == "21-50")
    ).astype(int)
    out["rf_away_receives_05_1_0_5m_active_21_50"] = (
        (out["side"] == "away")
        & (out["side_handicap_bucket"] == "side_receives_0.5_to_1")
        & (out["snapshot_bucket"] == "0-5m")
        & (out["activity_bucket"] == "21-50")
    ).astype(int)
    out["rf_away_weaker_receives_05_1_0_5m"] = (
        (out["side"] == "away")
        & (out["line_move"] == "side_weaker")
        & (out["side_handicap_bucket"] == "side_receives_0.5_to_1")
        & (out["snapshot_bucket"] == "0-5m")
    ).astype(int)
    out["rf_away_odds_08_09_receives_05_1_active_21_50"] = (
        (out["side"] == "away")
        & (out["side_odds_bucket"] == "0.80-0.90")
        & (out["side_handicap_bucket"] == "side_receives_0.5_to_1")
        & (out["activity_bucket"] == "21-50")
    ).astype(int)
    out["rf_any_strong_rule"] = (
        out[
            [
                "rf_side_weaker_0_5m_active_21_50",
                "rf_away_receives_05_1_active_21_50",
                "rf_away_receives_05_1_0_5m_active_21_50",
                "rf_away_weaker_receives_05_1_0_5m",
                "rf_away_odds_08_09_receives_05_1_active_21_50",
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)
    return out


def strong_rule_columns() -> list[str]:
    return [
        "rf_side_weaker_0_5m_active_21_50",
        "rf_away_receives_05_1_active_21_50",
        "rf_away_receives_05_1_0_5m_active_21_50",
        "rf_away_weaker_receives_05_1_0_5m",
        "rf_away_odds_08_09_receives_05_1_active_21_50",
        "rf_any_strong_rule",
    ]
