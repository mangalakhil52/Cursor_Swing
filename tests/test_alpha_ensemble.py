import pandas as pd
from src.alpha_ensemble import expanding_ensemble

def test_ensemble_scores_only_after_training_window():
    n=220; d=pd.DataFrame({'probability':[.5,.8]*(n//2),'structural_score':[60,80]*(n//2),'target_before_stop':[0,1]*(n//2)})
    x=expanding_ensemble(d,min_train=150,step=25)
    assert x.ensemble_probability.iloc[:150].isna().all()
    assert x.ensemble_probability.iloc[150:].notna().any()
