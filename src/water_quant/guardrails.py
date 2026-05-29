from __future__ import annotations

import pandas as pd

from .ml_baseline import add_month
from .rule_features import add_strong_rule_features


def summarize_bets(df: pd.DataFrame) -> dict[str, float | int]:
    profit = df["target_profit"] if not df.empty else pd.Series(dtype=float)
    return {
        "bets": int(len(df)),
        "profit": float(profit.sum()) if len(df) else 0.0,
        "roi": float(profit.mean()) if len(df) else 0.0,
        "win_rate": float((profit > 0).mean()) if len(df) else 0.0,
        "push_rate": float((profit == 0).mean()) if len(df) else 0.0,
        "lose_rate": float((profit < 0).mean()) if len(df) else 0.0,
    }


def guarded_rule_report(
    df: pd.DataFrame,
    *,
    rule_col: str,
    train_end_month: int = 202509,
    min_league_bets: int = 20,
    min_league_roi: float = 0.0,
    min_month_bets: int = 50,
    min_month_roi: float = 0.0,
) -> dict[str, pd.DataFrame]:
    data = add_month(add_strong_rule_features(df))
    hits = data[data[rule_col] == 1].copy()
    train_hits = hits[hits["month"] <= train_end_month].copy()
    test_hits = hits[hits["month"] > train_end_month].copy()

    league_train = (
        pd.DataFrame(
            [{"league": league, **summarize_bets(group)} for league, group in train_hits.groupby("league")]
        )
        if not train_hits.empty
        else pd.DataFrame()
    )
    whitelist = league_train[
        (league_train["bets"] >= min_league_bets) & (league_train["roi"] >= min_league_roi)
    ].copy()

    month_train = (
        pd.DataFrame(
            [{"month": int(month), **summarize_bets(group)} for month, group in train_hits.groupby("month")]
        )
        if not train_hits.empty
        else pd.DataFrame()
    )
    month_whitelist = month_train[
        (month_train["bets"] >= min_month_bets) & (month_train["roi"] >= min_month_roi)
    ].copy()

    scenarios = {
        "all": hits,
        "train_period": train_hits,
        "test_period": test_hits,
        "league_whitelist_all": hits[hits["league"].isin(whitelist["league"])],
        "league_whitelist_test": test_hits[test_hits["league"].isin(whitelist["league"])],
        "month_whitelist_all": hits[hits["month"].isin(month_whitelist["month"])],
        "month_whitelist_test": test_hits[test_hits["month"].isin(month_whitelist["month"])],
    }
    overall = pd.DataFrame(
        [{"scenario": name, **summarize_bets(part)} for name, part in scenarios.items()]
    )

    monthly = (
        pd.DataFrame(
            [{"month": int(month), **summarize_bets(group)} for month, group in hits.groupby("month")]
        )
        if not hits.empty
        else pd.DataFrame()
    )
    league_all = (
        pd.DataFrame(
            [{"league": league, **summarize_bets(group)} for league, group in hits.groupby("league")]
        )
        .sort_values(["roi", "bets"], ascending=[False, False])
        if not hits.empty
        else pd.DataFrame()
    )
    return {
        "overall": overall,
        "monthly": monthly,
        "league_train": league_train.sort_values(["roi", "bets"], ascending=[False, False])
        if not league_train.empty
        else league_train,
        "league_all": league_all,
        "whitelist": whitelist.sort_values(["roi", "bets"], ascending=[False, False]),
        "month_train": month_train.sort_values("month") if not month_train.empty else month_train,
        "month_whitelist": month_whitelist.sort_values("month")
        if not month_whitelist.empty
        else month_whitelist,
    }
