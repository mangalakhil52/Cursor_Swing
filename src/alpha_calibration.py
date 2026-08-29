"""Out-of-sample probability calibration diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd

def brier_score(probability, outcome):
    p=np.asarray(probability,dtype=float); y=np.asarray(outcome,dtype=float); m=np.isfinite(p)&np.isfinite(y)
    return float(np.mean((p[m]-y[m])**2)) if m.any() else float('nan')

def calibration_table(probability, outcome, bins=10):
    d=pd.DataFrame({'p':probability,'y':outcome}).dropna();
    if d.empty:return pd.DataFrame(columns=['bin','count','mean_pred','win_rate'])
    d['bin']=pd.cut(d.p,bins=np.linspace(0,1,bins+1),include_lowest=True)
    return d.groupby('bin',observed=True).agg(count=('y','size'),mean_pred=('p','mean'),win_rate=('y','mean')).reset_index()

def monotonic_calibration(probability,outcome):
    t=calibration_table(probability,outcome)
    if len(t)<2:return {'monotonic':False,'bins':len(t)}
    return {'monotonic':bool(t.win_rate.diff().dropna().ge(-.05).all()),'bins':len(t),'brier':brier_score(probability,outcome)}
