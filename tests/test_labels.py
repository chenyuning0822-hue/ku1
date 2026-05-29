from water_quant.labels import asian_handicap_profit


def test_flat_handicap_profit():
    assert asian_handicap_profit(1, 0, 0.9) == 0.9
    assert asian_handicap_profit(0, 0, 0.9) == 0.0
    assert asian_handicap_profit(-1, 0, 0.9) == -1.0


def test_quarter_handicap_profit():
    assert asian_handicap_profit(0, 0.25, 0.8) == 0.4
    assert asian_handicap_profit(0, -0.25, 0.8) == -0.5
    assert asian_handicap_profit(1, -0.75, 0.8) == 0.4
