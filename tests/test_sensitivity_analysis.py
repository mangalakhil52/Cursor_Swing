import pandas as pd
from src.sensitivity_analysis import evaluate_grid,plateau_score

def test_grid_has_expected_dimensions():
    d=pd.DataFrame({'probability':[.6,.7,.8]*20,'structural_score':[65,70,75]*20,'target_before_stop':[0,1,1]*20})
    x=evaluate_grid(d,probability_thresholds=(.6,.7),structural_thresholds=(65,70),stop_multipliers=(1,),target_multipliers=(1,))
    assert len(x)==4

def test_plateau_returns_boolean():
    x=plateau_score(pd.DataFrame({'expectancy_r':[.1,.11,.12,.13]}))
    assert isinstance(x['plateau'],bool)
