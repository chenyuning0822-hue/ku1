from __future__ import annotations

import pandas as pd

from .rules import _mask_for_rule, add_rule_buckets


def league_rule_breakdown(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    side: str,
    top_rules: int = 10,
    min_bets: int = 20,
) -> pd.DataFrame:
    if "league" not in df.columns:
        raise ValueError("Missing league column.")
    bucketed = add_rule_buckets(df, side=side)
    rows = []
    for rank, (_, rule_row) in enumerate(rules.head(top_rules).iterrows(), start=1):
        rule = str(rule_row["rule"])
        subset = bucketed[_mask_for_rule(bucketed, rule)]
        for league, group in subset.groupby("league", dropna=False):
            if len(group) < min_bets:
                continue
            profit = group["target_profit"]
            rows.append(
                {
                    "rule_rank": rank,
                    "rule": rule,
                    "league": league,
                    "bets": int(len(group)),
                    "profit": float(profit.sum()),
                    "roi": float(profit.mean()),
                    "win_rate": float((profit > 0).mean()),
                    "push_rate": float((profit == 0).mean()),
                    "lose_rate": float((profit < 0).mean()),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["rule_rank", "roi", "bets"], ascending=[True, False, False])


def league_overall_summary(df: pd.DataFrame, *, side: str, min_bets: int = 100) -> pd.DataFrame:
    target = f"target_{side}_profit"
    if "league" not in df.columns:
        raise ValueError("Missing league column.")
    rows = []
    for league, group in df.groupby("league", dropna=False):
        if len(group) < min_bets:
            continue
        profit = group[target]
        rows.append(
            {
                "league": league,
                "bets": int(len(group)),
                "profit": float(profit.sum()),
                "roi": float(profit.mean()),
                "win_rate": float((profit > 0).mean()),
                "push_rate": float((profit == 0).mean()),
                "lose_rate": float((profit < 0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["roi", "bets"], ascending=[False, False])


def rule_league_stability(league_breakdown: pd.DataFrame) -> pd.DataFrame:
    if league_breakdown.empty:
        return pd.DataFrame()
    grouped = league_breakdown.groupby(["rule_rank", "rule"])
    summary = (
        grouped
        .agg(
            leagues=("league", "nunique"),
            total_bets=("bets", "sum"),
            positive_leagues=("roi", lambda s: int((s > 0).sum())),
            min_league_roi=("roi", "min"),
            max_league_roi=("roi", "max"),
        )
        .reset_index()
    )
    totals = (
        grouped[["profit", "bets"]]
        .sum()
        .assign(weighted_roi=lambda x: x["profit"] / x["bets"])
        .drop(columns=["profit", "bets"])
        .reset_index()
    )
    return summary.merge(totals, on=["rule_rank", "rule"], how="left").sort_values(
        ["positive_leagues", "weighted_roi"], ascending=[False, False]
    )


def league_whitelist_candidates(
    league_breakdown: pd.DataFrame,
    *,
    min_bets: int = 30,
    min_roi: float = 0.03,
) -> pd.DataFrame:
    if league_breakdown.empty:
        return pd.DataFrame()
    candidates = league_breakdown[
        (league_breakdown["bets"] >= min_bets) & (league_breakdown["roi"] >= min_roi)
    ].copy()
    return candidates.sort_values(["roi", "bets"], ascending=[False, False])


def rule_month_league_grid(
    df: pd.DataFrame,
    rule: str,
    *,
    side: str,
    min_bets: int = 5,
) -> pd.DataFrame:
    if "date" not in df.columns or "league" not in df.columns:
        raise ValueError("Missing date or league column.")
    bucketed = add_rule_buckets(df, side=side)
    bucketed["month"] = pd.to_numeric(bucketed["date"], errors="coerce").astype("Int64") // 100
    subset = bucketed[_mask_for_rule(bucketed, rule)]
    rows = []
    for (month, league), group in subset.groupby(["month", "league"], dropna=False):
        if len(group) < min_bets:
            continue
        profit = group["target_profit"]
        rows.append(
            {
                "month": int(month),
                "league": league,
                "bets": int(len(group)),
                "profit": float(profit.sum()),
                "roi": float(profit.mean()),
                "win_rate": float((profit > 0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["month", "roi"], ascending=[True, False])
