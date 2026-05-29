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


def add_cumulative_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bet_no",
                "date",
                "match_id",
                "league",
                "side",
                "target_profit",
                "cum_profit",
                "running_peak",
                "drawdown",
            ]
        )
    sort_cols = [col for col in ["date", "match_id", "side"] if col in df.columns]
    out = df.sort_values(sort_cols).copy() if sort_cols else df.copy()
    out["bet_no"] = range(1, len(out) + 1)
    out["cum_profit"] = out["target_profit"].cumsum()
    out["running_peak"] = out["cum_profit"].cummax()
    out["drawdown"] = out["cum_profit"] - out["running_peak"]
    return out


def drawdown_summary(df: pd.DataFrame) -> dict[str, float | int]:
    curve = add_cumulative_stats(df)
    if curve.empty:
        return {"max_drawdown": 0.0, "final_profit": 0.0, "worst_bet_no": 0}
    worst_idx = curve["drawdown"].idxmin()
    return {
        "max_drawdown": float(curve["drawdown"].min()),
        "final_profit": float(curve["cum_profit"].iloc[-1]),
        "worst_bet_no": int(curve.loc[worst_idx, "bet_no"]),
    }


def league_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    rule_col: str,
    train_end_month: int = 202509,
    min_bet_values: tuple[int, ...] = (5, 10, 15, 20, 30, 50),
    min_roi_values: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02),
) -> pd.DataFrame:
    data = add_month(add_strong_rule_features(df))
    hits = data[data[rule_col] == 1].copy()
    train_hits = hits[hits["month"] <= train_end_month].copy()
    test_hits = hits[hits["month"] > train_end_month].copy()
    if train_hits.empty:
        return pd.DataFrame()

    league_train = pd.DataFrame(
        [{"league": league, **summarize_bets(group)} for league, group in train_hits.groupby("league")]
    )
    rows = []
    for min_bets in min_bet_values:
        for min_roi in min_roi_values:
            whitelist = league_train[
                (league_train["bets"] >= min_bets) & (league_train["roi"] >= min_roi)
            ]["league"]
            selected_test = test_hits[test_hits["league"].isin(whitelist)]
            summary = summarize_bets(selected_test)
            rows.append(
                {
                    "min_league_bets": min_bets,
                    "min_league_roi": min_roi,
                    "league_count": int(len(whitelist)),
                    **summary,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["bets", "roi", "league_count"], ascending=[False, False, False]
    )


def candidate_strategy_report(
    df: pd.DataFrame,
    *,
    rule_col: str,
    train_end_month: int = 202509,
) -> dict[str, pd.DataFrame]:
    data = add_month(add_strong_rule_features(df))
    hits = data[data[rule_col] == 1].copy()
    train_hits = hits[hits["month"] <= train_end_month].copy()
    test_hits = hits[hits["month"] > train_end_month].copy()

    scenarios = {
        "all": hits,
        "train_period": train_hits,
        "test_period": test_hits,
    }
    overall_rows = []
    for name, part in scenarios.items():
        overall_rows.append({"scenario": name, **summarize_bets(part), **drawdown_summary(part)})
    overall = pd.DataFrame(overall_rows)

    monthly = (
        pd.DataFrame(
            [
                {"month": int(month), **summarize_bets(group), **drawdown_summary(group)}
                for month, group in hits.groupby("month")
            ]
        ).sort_values("month")
        if not hits.empty
        else pd.DataFrame()
    )

    league = (
        pd.DataFrame(
            [{"league": league, **summarize_bets(group)} for league, group in hits.groupby("league")]
        ).sort_values(["bets", "roi"], ascending=[False, False])
        if not hits.empty
        else pd.DataFrame()
    )

    curve_cols = [
        "bet_no",
        "date",
        "match_id",
        "league",
        "home_team",
        "away_team",
        "side",
        "side_handicap_last",
        "side_odds_last",
        "target_profit",
        "cum_profit",
        "drawdown",
    ]
    curve = add_cumulative_stats(hits)
    curve = curve[[col for col in curve_cols if col in curve.columns]] if not curve.empty else curve

    return {
        "overall": overall,
        "monthly": monthly,
        "league": league,
        "league_threshold_sensitivity": league_threshold_sensitivity(
            df, rule_col=rule_col, train_end_month=train_end_month
        ),
        "bet_curve": curve,
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
