import pandas as pd
import numpy as np
from src.research_engine import evaluate_symbol, summarize

def test_no_lookahead_and_output_columns():
    n=180; idx=pd.date_range('2020-01-01',periods=n,freq='B'); close=pd.Series(100*np.exp(np.cumsum(np.full(n,.001))),index=idx); d=pd.DataFrame({'open':close*0.999,'high':close*1.01,'low':close*.99,'close':close,'volume':1000000},index=idx)
    out=evaluate_symbol('TEST',d)
    assert not out.empty
    assert {'execution_date','score','mfe_pct','mae_pct','r_multiple','fwd5_pct','fwd10_pct'}.issubset(out.columns)
    assert out.execution_date.min()>out.signal_date.min()

def test_summary():
    d=pd.DataFrame({'r_multiple':[1.7,-1,1.7,-1,0.2],'target_hit':[1,0,1,0,0],'stop_hit':[0,1,0,1,0],'mfe_pct':[10,1,8,1,2],'mae_pct':[-1,-5,-1,-5,-2]})
    s=summarize(d)
    assert s['trades']==5
    assert s['win_rate']>0.5
    assert s['profit_factor']>1
