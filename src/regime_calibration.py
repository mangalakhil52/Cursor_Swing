"""Evaluate probability calibration separately by market regime."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

def by_regime(df:pd.DataFrame, regime_col='regime', y_col='target_before_stop', p_col='probability', min_obs=20):
    rows=[]
    for regime,g in df.groupby(regime_col,dropna=False):
        y=pd.to_numeric(g[y_col],errors='coerce'); p=pd.to_numeric(g[p_col],errors='coerce'); x=pd.DataFrame({'y':y,'p':p}).dropna()
        if len(x)<min_obs: continue
        rows.append({'regime':str(regime),'observations':len(x),'brier':float(brier_score_loss(x.y,x.p)),'actual_rate':float(x.y.mean()),'predicted_rate':float(x.p.mean()),'calibration_gap':float(abs(x.y.mean()-x.p.mean()))})
    return pd.DataFrame(rows)

def gate(regime_rows:pd.DataFrame,max_gap=.12,min_obs=20):
    if regime_rows.empty:return {'pass':False,'reason':'no_regime_with_enough_observations'}
    bad=regime_rows[(regime_rows.observations>=min_obs)&(regime_rows.calibration_gap>max_gap)]
    return {'pass':bool(bad.empty),'bad_regimes':bad.regime.astype(str).tolist(),'max_gap':float(regime_rows.calibration_gap.max())}
