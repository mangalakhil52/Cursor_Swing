from src.ensemble_engine import ensemble_score, decision

def test_bull_weights_alpha_more():
    assert ensemble_score(80,.80,'BULL') > ensemble_score(80,.80,'SIDEWAYS')

def test_probability_floor_rejects():
    x=decision(100,.59,'BULL')
    assert x['pass'] is False

def test_score_is_bounded():
    x=ensemble_score(100,1,'STRONG_BULL')
    assert 0<=x<=100
