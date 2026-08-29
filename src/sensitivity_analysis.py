"""Sensitivity analysis for detecting fragile strategy parameter optima."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

def evaluate_grid(df:pd.DataFrame, probability_thresholds=(.55,.60,.65,.70,.75), structural_thresholds=(60,64,68,72,76), stop_multipliers=(.9,1.,1.1), target_multipliers=(.9,1.,1.1)):
    rows=[]
    for p,s,sm,tm in itertools.product(probability_thresholds,structural_thresholds,stop_multipliers,target_multipliers):
        take=(df['probability']>=p)&(df['structural_score']>=s)
        y=pd.to_numeric(df.loc[take,'target_before_stop'],errors='coerce').dropna()
        r=np.where(y.astype(int).eq(1),2.*tm,-1.*sm)
        rows.append({'probability_threshold':p,'structural_threshold':s,'stop_multiplier':sm,'target_multiplier':tm,'trades':len(r),'expectancy_r':float(r.mean()) if len(r) else np.nan,'total_r':float(r.sum()) if len(r) else np.nan})
    return pd.DataFrame(rows)

def plateau_score(grid:pd.DataFrame)->dict:
    x=grid.dropna(subset=['expectancy_r']).copy()
    if x.empty:return {'best_expectancy_r':np.nan,'top_quartile_share':0.,'plateau':False}
    best=float(x.expectancy_r.max()); q=float(x.expectancy_r.quantile(.75)); share=float((x.expectancy_r>=q).mean())
    return {'best_expectancy_r':best,'top_quartile_share':share,'plateau':bool(share>=.15)}
