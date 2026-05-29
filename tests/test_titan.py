from water_quant.titan import parse_final_score, parse_handicap_text


def test_parse_final_score():
    assert parse_final_score("2-1") == (2.0, 1.0)
    assert parse_final_score(" 0 - 0 ") == (0.0, 0.0)
    assert parse_final_score("推迟") == (None, None)


def test_parse_handicap_text_home_gives():
    assert parse_handicap_text("平手") == 0.0
    assert parse_handicap_text("平/半") == -0.25
    assert parse_handicap_text("半球") == -0.5
    assert parse_handicap_text("一球/球半") == -1.25


def test_parse_handicap_text_home_receives():
    assert parse_handicap_text("*平/半") == -0.25
    assert parse_handicap_text("受让平/半") == 0.25
    assert parse_handicap_text("受让一球/球半") == 1.25
