from water_quant.labels import asian_handicap_profit
from water_quant.labels import attach_home_handicap_target
import pandas as pd


def test_flat_handicap_profit():
    assert asian_handicap_profit(1, 0, 0.9) == 0.9
    assert asian_handicap_profit(0, 0, 0.9) == 0.0
    assert asian_handicap_profit(-1, 0, 0.9) == -1.0


def test_quarter_handicap_profit():
    assert asian_handicap_profit(0, 0.25, 0.8) == 0.4
    assert asian_handicap_profit(0, -0.25, 0.8) == -0.5
    assert asian_handicap_profit(1, -0.75, 0.8) == 0.4


def test_attach_home_and_away_targets():
    features = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "handicap_last": -0.5,
                "home_odds_last": 0.9,
                "away_odds_last": 0.95,
            }
        ]
    )
    results = pd.DataFrame([{"match_id": "m1", "home_score": 1, "away_score": 1}])
    labeled = attach_home_handicap_target(features, results)
    assert labeled.loc[0, "target_home_profit"] == -1.0
    assert labeled.loc[0, "target_away_profit"] == 0.95
