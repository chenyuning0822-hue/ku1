from __future__ import annotations

import pandas as pd


def make_signals(
    df: pd.DataFrame,
    *,
    prediction_col: str = "predicted_home_profit",
    threshold: float = 0.03,
) -> pd.DataFrame:
    signals = df.copy()
    signals["bet_home"] = signals[prediction_col] > threshold
    signals["strategy_profit"] = signals["target_home_profit"].where(
        signals["bet_home"], 0.0
    )
    return signals


def summarize_backtest(signals: pd.DataFrame) -> pd.DataFrame:
    bets = signals[signals["bet_home"]].copy()
    total_profit = float(bets["strategy_profit"].sum()) if not bets.empty else 0.0
    bet_count = int(len(bets))
    turnover = float(bet_count)
    roi = total_profit / turnover if turnover else 0.0
    hit_rate = float((bets["strategy_profit"] > 0).mean()) if bet_count else 0.0
    summary = {
        "matches": int(len(signals)),
        "bets": bet_count,
        "total_profit": total_profit,
        "turnover": turnover,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_profit_per_bet": roi,
    }
    return pd.DataFrame([summary])

