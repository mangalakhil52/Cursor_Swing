import pandas as pd
from src.cross_sectional_alpha import walk_forward_score

def test_future_dates_are_not_used():
    dates=pd.date_range('2026-01-01',periods=35,freq='D')
    rows=[]
    for d in dates:
        rows.extend([{'date':d,'symbol':'A','probability':.8,'structural_score':80,'target_before_stop':1},{'date':d,'symbol':'B','probability':.6,'structural_score':60,'target_before_stop':0}])
    x=walk_forward_score(pd.DataFrame(rows),min_train_dates=30)
    assert x[x.date<dates[30]].learned_alpha.isna().all()
    assert x[x.date>=dates[30]].learned_alpha.notna().any()
