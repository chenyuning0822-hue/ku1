from __future__ import annotations

import pandas as pd

from .rules import add_rule_buckets


METADATA_COLUMNS = [
    "match_id",
    "date",
    "league",
    "home_team",
    "away_team",
    "start_time_text",
    "final_score",
    "home_score",
    "away_score",
]


def build_side_dataset(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ["home", "away"]:
        side_df = add_rule_buckets(features, side=side).copy()
        side_df["side"] = side
        side_df["opponent"] = "away" if side == "home" else "home"
        side_df["is_home_side"] = side == "home"
        side_df["side_score"] = side_df["home_score"] if side == "home" else side_df["away_score"]
        side_df["opponent_score"] = (
            side_df["away_score"] if side == "home" else side_df["home_score"]
        )
        side_df["side_margin"] = side_df["side_score"] - side_df["opponent_score"]

        keep = [col for col in METADATA_COLUMNS if col in side_df.columns]
        keep += [
            "side",
            "opponent",
            "is_home_side",
            "side_score",
            "opponent_score",
            "side_margin",
            "target_profit",
            "side_handicap_first",
            "side_handicap_last",
            "side_handicap_change",
            "side_odds_first",
            "side_odds_last",
            "side_odds_change",
            "line_move",
            "odds_move",
            "side_odds_bucket",
            "side_handicap_bucket",
            "snapshot_bucket",
            "activity_bucket",
        ]
        for col in [
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
        ]:
            if col in side_df.columns:
                keep.append(col)
        rows.append(side_df[keep])

    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["date", "match_id", "side"], kind="stable").reset_index(drop=True)


def side_dataset_summary(side_df: pd.DataFrame) -> pd.DataFrame:
    return (
        side_df.groupby("side")
        .agg(
            samples=("target_profit", "size"),
            mean_profit=("target_profit", "mean"),
            win_rate=("target_profit", lambda s: (s > 0).mean()),
            push_rate=("target_profit", lambda s: (s == 0).mean()),
            lose_rate=("target_profit", lambda s: (s < 0).mean()),
        )
        .reset_index()
    )
