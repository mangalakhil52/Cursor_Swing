import pandas as pd
import numpy as np
from src.risk_parity import risk_parity_weights,portfolio_stats

def test_weights_sum_to_one():
    r=pd.DataFrame({'A':np.arange(100)/100,'B':np.arange(100)[::-1]/100,'C':np.sin(np.arange(100))})
    w=risk_parity_weights(r,['A','B','C']); assert abs(sum(w.values())-1)<1e-9

def test_stats_are_finite():
    r=pd.DataFrame({'A':np.arange(100)/100,'B':np.arange(100)[::-1]/100}); w=risk_parity_weights(r,['A','B']); s=portfolio_stats(r,w); assert np.isfinite(s['annualized_vol']) and s['effective_positions']>=1
