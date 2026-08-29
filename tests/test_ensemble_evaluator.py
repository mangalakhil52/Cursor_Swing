import pandas as pd
from src.ensemble_evaluator import evaluate

def test_evaluator_returns_both_strategies():
    d=pd.DataFrame({'structural_score':[80,70,60,90],'probability':[.8,.7,.5,.9],'regime':['BULL']*4,'target_before_stop':[1,0,1,1]})
    out=evaluate(d)
    assert set(out.strategy)=={'BASE','ENSEMBLE'}
    assert len(out)==2
