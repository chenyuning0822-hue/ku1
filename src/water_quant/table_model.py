from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_GROUPS = [
    ("line_move", "snapshot_bucket", "activity_bucket"),
    ("odds_move", "side_handicap_bucket", "activity_bucket"),
    ("side_odds_bucket", "side_handicap_bucket", "activity_bucket"),
    ("line_move", "side_handicap_bucket", "snapshot_bucket"),
    ("side", "side_handicap_bucket", "activity_bucket"),
]


@dataclass
class TableRule:
    columns: tuple[str, ...]
    table: pd.DataFrame
    global_mean: float
    smoothing: float
    min_count: int


class SmoothedTableModel:
    def __init__(
        self,
        groups: list[tuple[str, ...]] | None = None,
        *,
        smoothing: float = 200.0,
        min_count: int = 50,
    ) -> None:
        self.groups = groups or DEFAULT_GROUPS
        self.smoothing = smoothing
        self.min_count = min_count
        self.global_mean = 0.0
        self.rules: list[TableRule] = []

    def fit(self, df: pd.DataFrame) -> "SmoothedTableModel":
        self.global_mean = float(df["target_profit"].mean())
        self.rules = []
        for columns in self.groups:
            existing = tuple(col for col in columns if col in df.columns)
            if len(existing) != len(columns):
                continue
            grouped = (
                df.groupby(list(existing), dropna=False)["target_profit"]
                .agg(["count", "mean"])
                .reset_index()
            )
            grouped = grouped[grouped["count"] >= self.min_count].copy()
            if grouped.empty:
                continue
            grouped["prediction"] = (
                grouped["count"] * grouped["mean"] + self.smoothing * self.global_mean
            ) / (grouped["count"] + self.smoothing)
            self.rules.append(
                TableRule(
                    columns=existing,
                    table=grouped[list(existing) + ["prediction"]],
                    global_mean=self.global_mean,
                    smoothing=self.smoothing,
                    min_count=self.min_count,
                )
            )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pred_sum = pd.Series(0.0, index=df.index)
        pred_count = pd.Series(0.0, index=df.index)
        for rule in self.rules:
            table = rule.table.rename(columns={"prediction": "_rule_prediction"})
            merged = df[list(rule.columns)].merge(
                table, on=list(rule.columns), how="left", sort=False
            )
            values = merged["_rule_prediction"].set_axis(df.index)
            mask = values.notna()
            pred_sum.loc[mask] += values.loc[mask]
            pred_count.loc[mask] += 1
        predictions = pred_sum / pred_count.replace(0, pd.NA)
        return predictions.fillna(self.global_mean).astype(float)

    def export_tables(self) -> dict[str, pd.DataFrame]:
        return {"__".join(rule.columns): rule.table.copy() for rule in self.rules}
