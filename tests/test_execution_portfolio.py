import pandas as pd
from src.execution_portfolio import simulate_candidates

def test_integrated_execution_uses_ohlc():
    candidates=pd.DataFrame({'symbol':['A'],'entry_date':pd.to_datetime(['2026-01-01']),'score':[90],'direction':['LONG'],'entry_price':[100.],'stop_price':[95.],'target1':[105.],'target2':[110.],'probability':[.8],'vol_annual':[.2]})
    bars=pd.DataFrame({'open':[100.,105.],'high':[106.,112.],'low':[99.,104.],'close':[105.,111.]},index=pd.to_datetime(['2026-01-02','2026-01-05']))
    a,r=simulate_candidates(candidates,{'A':bars})
    assert len(a)==1 and a.iloc[0].status=='TARGET2_HIT' and len(r)==0
