from __future__ import annotations

from itertools import combinations

import pandas as pd


def add_rule_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["handicap_change"] = out["handicap_last"] - out["handicap_first"]
    out["home_odds_change"] = out["home_odds_last"] - out["home_odds_first"]

    out["line_move"] = "same"
    out.loc[out["handicap_change"] < 0, "line_move"] = "home_stronger"
    out.loc[out["handicap_change"] > 0, "line_move"] = "home_weaker"

    out["odds_move"] = "same"
    out.loc[out["home_odds_change"] < 0, "odds_move"] = "home_odds_down"
    out.loc[out["home_odds_change"] > 0, "odds_move"] = "home_odds_up"

    out["home_odds_bucket"] = pd.cut(
        out["home_odds_last"],
        bins=[0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 10],
        labels=["<=0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00", "1.00-1.10", "1.10-1.30", ">1.30"],
        include_lowest=True,
    ).astype(str)

    out["handicap_bucket"] = pd.cut(
        out["handicap_last"],
        bins=[-10, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 10],
        labels=[
            "home_gives_1.5+",
            "home_gives_1_to_1.5",
            "home_gives_0.5_to_1",
            "home_gives_0_to_0.5",
            "home_receives_0_to_0.5",
            "home_receives_0.5_to_1",
            "home_receives_1_to_1.5",
            "home_receives_1.5+",
        ],
        include_lowest=True,
    ).astype(str)

    out["snapshot_bucket"] = pd.cut(
        out["last_minute_to_start"],
        bins=[-0.001, 5, 10, 30, 60, 180, 720, 100000],
        labels=["0-5m", "5-10m", "10-30m", "30-60m", "1-3h", "3-12h", "12h+"],
        include_lowest=True,
    ).astype(str)

    if "row_count" in out.columns:
        out["activity_bucket"] = pd.cut(
            out["row_count"],
            bins=[0, 3, 8, 20, 50, 10000],
            labels=["<=3", "4-8", "9-20", "21-50", "50+"],
            include_lowest=True,
        ).astype(str)
    return out


def _summarize_group(group: pd.DataFrame, rule_columns: tuple[str, ...]) -> dict[str, object]:
    profit = group["target_home_profit"]
    return {
        "rule": " & ".join(f"{col}={group[col].iloc[0]}" for col in rule_columns),
        "columns": ",".join(rule_columns),
        "bets": int(len(group)),
        "profit": float(profit.sum()),
        "roi": float(profit.mean()),
        "win_rate": float((profit > 0).mean()),
        "push_rate": float((profit == 0).mean()),
        "lose_rate": float((profit < 0).mean()),
        "avg_home_odds": float(group["home_odds_last"].mean()),
        "avg_handicap": float(group["handicap_last"].mean()),
    }


def search_rule_segments(
    df: pd.DataFrame,
    *,
    min_bets: int = 80,
    max_depth: int = 3,
) -> pd.DataFrame:
    bucketed = add_rule_buckets(df)
    candidate_columns = [
        "line_move",
        "odds_move",
        "home_odds_bucket",
        "handicap_bucket",
        "snapshot_bucket",
        "activity_bucket",
    ]
    candidate_columns = [col for col in candidate_columns if col in bucketed.columns]

    rows: list[dict[str, object]] = []
    for depth in range(1, max_depth + 1):
        for cols in combinations(candidate_columns, depth):
            grouped = bucketed.groupby(list(cols), observed=False, dropna=False)
            for _, group in grouped:
                if len(group) < min_bets:
                    continue
                rows.append(_summarize_group(group, cols))

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["edge_vs_all"] = result["roi"] - float(df["target_home_profit"].mean())
    return result.sort_values(["roi", "bets"], ascending=[False, False]).reset_index(drop=True)


def _mask_for_rule(bucketed: pd.DataFrame, rule: str) -> pd.Series:
    mask = pd.Series(True, index=bucketed.index)
    for part in rule.split(" & "):
        column, value = part.split("=", 1)
        mask &= bucketed[column].astype(str) == value
    return mask


def validate_rules_by_period(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    date_col: str = "date",
    split_date: int | None = None,
    top_n: int = 100,
) -> pd.DataFrame:
    if date_col not in df.columns or rules.empty:
        return pd.DataFrame()
    bucketed = add_rule_buckets(df)
    dates = pd.to_numeric(bucketed[date_col], errors="coerce")
    if split_date is None:
        split_date = int(dates.median())

    rows = []
    for _, rule_row in rules.head(top_n).iterrows():
        mask = _mask_for_rule(bucketed, str(rule_row["rule"]))
        early = bucketed[mask & (dates <= split_date)]
        late = bucketed[mask & (dates > split_date)]
        rows.append(
            {
                "rule": rule_row["rule"],
                "all_bets": int(rule_row["bets"]),
                "all_roi": float(rule_row["roi"]),
                "early_bets": int(len(early)),
                "early_roi": float(early["target_home_profit"].mean())
                if len(early)
                else None,
                "late_bets": int(len(late)),
                "late_roi": float(late["target_home_profit"].mean())
                if len(late)
                else None,
                "min_period_roi": min(
                    float(early["target_home_profit"].mean()) if len(early) else -999,
                    float(late["target_home_profit"].mean()) if len(late) else -999,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["min_period_roi", "all_roi"], ascending=[False, False]
    )


def apply_rule_table(df: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame()
    bucketed = add_rule_buckets(df)
    rows = []
    for _, rule_row in rules.iterrows():
        mask = _mask_for_rule(bucketed, str(rule_row["rule"]))
        subset = bucketed[mask]
        profit = subset["target_home_profit"] if len(subset) else pd.Series(dtype=float)
        rows.append(
            {
                "rule": rule_row["rule"],
                "source_bets": int(rule_row["bets"]) if "bets" in rule_row else None,
                "source_roi": float(rule_row["roi"]) if "roi" in rule_row else None,
                "test_bets": int(len(subset)),
                "test_profit": float(profit.sum()) if len(subset) else 0.0,
                "test_roi": float(profit.mean()) if len(subset) else None,
                "test_win_rate": float((profit > 0).mean()) if len(subset) else None,
                "test_push_rate": float((profit == 0).mean()) if len(subset) else None,
                "test_lose_rate": float((profit < 0).mean()) if len(subset) else None,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(["test_roi", "test_bets"], ascending=[False, False])


def apply_rules_by_period(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    period_col: str = "month",
) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame()
    bucketed = add_rule_buckets(df)
    if period_col not in bucketed.columns:
        raise ValueError(f"Missing period column: {period_col}")

    rows = []
    for _, rule_row in rules.iterrows():
        mask = _mask_for_rule(bucketed, str(rule_row["rule"]))
        subset = bucketed[mask]
        for period, group in subset.groupby(period_col, dropna=False):
            profit = group["target_home_profit"]
            rows.append(
                {
                    "rule": rule_row["rule"],
                    "period": period,
                    "bets": int(len(group)),
                    "profit": float(profit.sum()),
                    "roi": float(profit.mean()) if len(group) else None,
                    "win_rate": float((profit > 0).mean()) if len(group) else None,
                }
            )
    return pd.DataFrame(rows)
