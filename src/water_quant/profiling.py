from __future__ import annotations

import pandas as pd


def _profit_summary(df: pd.DataFrame) -> pd.DataFrame:
    bets = len(df)
    return pd.DataFrame(
        [
            {
                "samples": bets,
                "mean_profit": float(df["target_home_profit"].mean()),
                "win_rate": float((df["target_home_profit"] > 0).mean()),
                "push_rate": float((df["target_home_profit"] == 0).mean()),
                "lose_rate": float((df["target_home_profit"] < 0).mean()),
                "min_profit": float(df["target_home_profit"].min()),
                "max_profit": float(df["target_home_profit"].max()),
            }
        ]
    )


def _bucket_series(series: pd.Series, bins: list[float]) -> pd.Categorical:
    return pd.cut(series, bins=bins, include_lowest=True, duplicates="drop")


def profile_features(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {
        "target_home_profit",
        "handicap_first",
        "handicap_last",
        "home_odds_first",
        "home_odds_last",
        "away_odds_first",
        "away_odds_last",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    out: dict[str, pd.DataFrame] = {}
    out["profit_summary"] = _profit_summary(df)

    out["target_distribution"] = (
        df["target_home_profit"]
        .round(2)
        .value_counts()
        .sort_index()
        .rename_axis("target_home_profit")
        .reset_index(name="samples")
    )

    if "date" in df.columns:
        out["by_date"] = (
            df.groupby("date", dropna=False)
            .agg(
                samples=("target_home_profit", "size"),
                mean_profit=("target_home_profit", "mean"),
                win_rate=("target_home_profit", lambda s: (s > 0).mean()),
            )
            .reset_index()
            .sort_values("date")
        )

    if "league" in df.columns:
        out["by_league"] = (
            df.groupby("league", dropna=False)
            .agg(
                samples=("target_home_profit", "size"),
                mean_profit=("target_home_profit", "mean"),
                win_rate=("target_home_profit", lambda s: (s > 0).mean()),
            )
            .reset_index()
            .sort_values(["samples", "mean_profit"], ascending=[False, False])
        )

    handicap = df.copy()
    handicap["handicap_change"] = handicap["handicap_last"] - handicap["handicap_first"]
    handicap["home_odds_change"] = handicap["home_odds_last"] - handicap["home_odds_first"]
    handicap["away_odds_change"] = handicap["away_odds_last"] - handicap["away_odds_first"]
    handicap["handicap_bucket"] = handicap["handicap_last"].round(2)
    out["by_handicap"] = (
        handicap.groupby("handicap_bucket", dropna=False)
        .agg(
            samples=("target_home_profit", "size"),
            mean_profit=("target_home_profit", "mean"),
            win_rate=("target_home_profit", lambda s: (s > 0).mean()),
            avg_home_odds=("home_odds_last", "mean"),
        )
        .reset_index()
        .sort_values("handicap_bucket")
    )

    movement = handicap.copy()
    movement["handicap_move"] = pd.Series("same", index=movement.index)
    movement.loc[movement["handicap_change"] > 0, "handicap_move"] = "home_less_disadvantaged"
    movement.loc[movement["handicap_change"] < 0, "handicap_move"] = "home_more_disadvantaged"
    movement["home_odds_move"] = pd.Series("same", index=movement.index)
    movement.loc[movement["home_odds_change"] > 0, "home_odds_move"] = "home_odds_up"
    movement.loc[movement["home_odds_change"] < 0, "home_odds_move"] = "home_odds_down"
    out["by_movement"] = (
        movement.groupby(["handicap_move", "home_odds_move"], dropna=False)
        .agg(
            samples=("target_home_profit", "size"),
            mean_profit=("target_home_profit", "mean"),
            win_rate=("target_home_profit", lambda s: (s > 0).mean()),
            avg_handicap_change=("handicap_change", "mean"),
            avg_home_odds_change=("home_odds_change", "mean"),
        )
        .reset_index()
        .sort_values(["mean_profit", "samples"], ascending=[False, False])
    )

    odds_bins = [0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 3]
    odds = df.copy()
    odds["home_odds_bucket"] = _bucket_series(odds["home_odds_last"], odds_bins)
    out["by_home_odds_bucket"] = (
        odds.groupby("home_odds_bucket", observed=False)
        .agg(
            samples=("target_home_profit", "size"),
            mean_profit=("target_home_profit", "mean"),
            win_rate=("target_home_profit", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )

    if "last_minute_to_start" in df.columns:
        timing = df.copy()
        timing["last_snapshot_bucket"] = _bucket_series(
            timing["last_minute_to_start"],
            [0, 5, 10, 30, 60, 180, 720, 1440, 100000],
        )
        out["by_last_snapshot_time"] = (
            timing.groupby("last_snapshot_bucket", observed=False)
            .agg(
                samples=("target_home_profit", "size"),
                mean_profit=("target_home_profit", "mean"),
                win_rate=("target_home_profit", lambda s: (s > 0).mean()),
                avg_last_minute=("last_minute_to_start", "mean"),
            )
            .reset_index()
        )

    numeric = df.select_dtypes("number")
    out["missing_numeric"] = (
        numeric.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return out


def baseline_rules(df: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "always_home": pd.Series(True, index=df.index),
        "home_odds_down": df["home_odds_last"] < df["home_odds_first"],
        "home_odds_up": df["home_odds_last"] > df["home_odds_first"],
        "handicap_home_stronger": df["handicap_last"] < df["handicap_first"],
        "handicap_home_weaker": df["handicap_last"] > df["handicap_first"],
        "low_home_odds": df["home_odds_last"] <= 0.85,
        "high_home_odds": df["home_odds_last"] >= 1.0,
        "odds_down_no_line_move": (df["home_odds_last"] < df["home_odds_first"])
        & (df["handicap_last"] == df["handicap_first"]),
        "odds_up_no_line_move": (df["home_odds_last"] > df["home_odds_first"])
        & (df["handicap_last"] == df["handicap_first"]),
    }
    rows = []
    for name, mask in rules.items():
        subset = df[mask.fillna(False)]
        rows.append(
            {
                "rule": name,
                "bets": int(len(subset)),
                "coverage": float(len(subset) / len(df)) if len(df) else 0.0,
                "profit": float(subset["target_home_profit"].sum()) if len(subset) else 0.0,
                "roi": float(subset["target_home_profit"].mean()) if len(subset) else 0.0,
                "win_rate": float((subset["target_home_profit"] > 0).mean())
                if len(subset)
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["roi", "bets"], ascending=[False, False])
