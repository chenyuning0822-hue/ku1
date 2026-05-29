from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline


ID_COLUMNS = {"match_id"}
TARGET_COLUMNS = {
    "target_home_profit",
    "target_away_profit",
    "home_score",
    "away_score",
    "home_margin",
}
METADATA_COLUMNS = {
    "date",
    "league",
    "home_team",
    "away_team",
    "start_time_text",
    "final_score",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = ID_COLUMNS | TARGET_COLUMNS | METADATA_COLUMNS
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def make_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=10,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_profit_model(df: pd.DataFrame) -> tuple[Pipeline, list[str], dict[str, float]]:
    cols = feature_columns(df)
    if not cols:
        raise ValueError("No numeric feature columns available for training.")
    X = df[cols]
    y = df["target_home_profit"]
    model = make_model()
    metrics: dict[str, float] = {}
    if len(df) >= 20:
        splits = min(5, max(2, len(df) // 20))
        cv = TimeSeriesSplit(n_splits=splits)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        metrics["cv_mae"] = float(-scores.mean())
    model.fit(X, y)
    pred = model.predict(X)
    metrics["train_mae"] = float(mean_absolute_error(y, pred))
    return model, cols, metrics
