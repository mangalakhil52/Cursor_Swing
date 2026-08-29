import pandas as pd
from src.execution_path import simulate_trade

def test_stop_first_when_both_touched():
    d=pd.DataFrame({'open':[100],'high':[110],'low':[90],'close':[105]},index=pd.to_datetime(['2026-01-02']))
    x=simulate_trade(d,'LONG',100,95,105,110)
    assert x['status']=='STOPPED'

def test_gap_stop_uses_open():
    d=pd.DataFrame({'open':[90],'high':[95],'low':[88],'close':[92]},index=pd.to_datetime(['2026-01-02']))
    x=simulate_trade(d,'LONG',100,95,110,115)
    assert x['exit_price']==90

def test_target_then_breakeven_then_target2():
    d=pd.DataFrame({'open':[100,105],'high':[106,112],'low':[99,104],'close':[105,111]},index=pd.to_datetime(['2026-01-02','2026-01-05']))
    x=simulate_trade(d,'LONG',100,95,105,110)
    assert x['status']=='TARGET2_HIT' and x['r_multiple']>1
