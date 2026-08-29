import pandas as pd
from src.regime_calibration import by_regime,gate

def test_regime_calibration():
    d=pd.DataFrame({'regime':['BULL']*20,'target_before_stop':[1]*10+[0]*10,'probability':[.5]*20})
    x=by_regime(d); assert len(x)==1 and x.iloc[0].calibration_gap==0

def test_bad_regime_fails_gate():
    x=pd.DataFrame({'regime':['BULL'],'observations':[20],'calibration_gap':[.2],'brier':[.2],'actual_rate':[.5],'predicted_rate':[.7]})
    assert gate(x)['pass'] is False
