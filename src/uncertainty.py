"""Calibration and uncertainty diagnostics for model probabilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

def calibration_bins(y,p,bins=10):
    d=pd.DataFrame({'y':pd.Series(y).astype(float),'p':pd.Series(p).astype(float)}).dropna(); d['bin']=pd.cut(d.p,np.linspace(0,1,bins+1),include_lowest=True)
    return d.groupby('bin',observed=True).agg(observations=('y','size'),predicted=('p','mean'),actual=('y','mean')).reset_index()

def uncertainty(y,p,bins=10):
    y=pd.Series(y); p=pd.Series(p); d=pd.DataFrame({'y':y,'p':p}).dropna()
    if d.empty:return {'brier':np.nan,'calibration_gap':np.nan,'coverage':0.}
    c=calibration_bins(d.y,d.p,bins); gap=float(np.average(abs(c.predicted-c.actual),weights=c.observations)) if len(c) else np.nan
    return {'brier':float(brier_score_loss(d.y,d.p)),'calibration_gap':gap,'coverage':float(len(d)/max(len(y),1))}

def confidence_gate(probability, calibration_gap, *, min_probability=.62, max_calibration_gap=.10):
    p=float(probability); return bool(p>=min_probability and float(calibration_gap)<=max_calibration_gap)
