import pandas as pd
from src.strategy_compare import simulate

def test_simulator_counts_target_and_stop():
    d=pd.DataFrame({'probability':[.8,.9,.4],'target_before_stop':[1,0,1]})
    r=simulate(d,'probability',.6,0)
    assert r['trades']==2
    assert r['total_r']==1
    assert r['win_rate']==.5
