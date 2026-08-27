"""Parameter-perturbation and execution-stress diagnostics.

Purpose: detect brittle parameter choices without selecting the best perturbation on OOS data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def perturbation_grid(base:dict)->list[dict]:
    grids={
      'min_score':[base.get('min_score',68)-4,base.get('min_score',68),base.get('min_score',68)+4],
      'target_r':[max(1.2,base.get('target_r',1.7)-.2),base.get('target_r',1.7),base.get('target_r',1.7)+.2],
      'stop_pct':[max(.02,base.get('stop_pct',.05)-.01),base.get('stop_pct',.05),min(.08,base.get('stop_pct',.05)+.01)]}
    import itertools
    keys=list(grids); return [dict(zip(keys,v)) for v in itertools.product(*(grids[k] for k in keys))]


def stability_summary(metrics:pd.DataFrame)->dict:
    if metrics.empty:return {'stable':False,'reason':'no scenarios'}
    profitable=(metrics.avg_r>0).mean(); positive_pf=(metrics.profit_factor>=1).mean()
    return {'scenarios':len(metrics),'positive_expectancy_share':float(profitable),'pf_ge_1_share':float(positive_pf),'stable':bool(profitable>=.70 and positive_pf>=.70),'worst_avg_r':float(metrics.avg_r.min()),'median_avg_r':float(metrics.avg_r.median())}


def bootstrap_ci(r, n=5000, seed=42):
    x=np.asarray(r,dtype=float); x=x[np.isfinite(x)]
    if len(x)<20:return (float('nan'),float('nan'))
    rng=np.random.default_rng(seed); means=np.mean(rng.choice(x,(n,len(x)),replace=True),axis=1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))
