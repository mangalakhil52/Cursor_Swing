"""Robust economic metrics for OOS strategy promotion."""
from __future__ import annotations
import numpy as np
import pandas as pd

def metrics(r: pd.Series, periods_per_year: int = 252) -> dict:
    x=pd.to_numeric(r,errors='coerce').dropna().astype(float)
    if x.empty:return {'trades':0,'expectancy_r':0.,'profit_factor':0.,'max_drawdown_r':0.,'sharpe':0.,'sortino':0.,'calmar':0.,'tail_5pct_r':0.}
    eq=x.cumsum(); peak=eq.cummax(); dd=eq-peak
    mu=float(x.mean()); sd=float(x.std(ddof=1)) if len(x)>1 else 0.
    downside=x[x<0]; dsd=float(downside.std(ddof=1)) if len(downside)>1 else 0.
    wins=float(x[x>0].sum()); losses=float(-x[x<0].sum())
    ann=mu*np.sqrt(periods_per_year)
    return {'trades':int(len(x)),'expectancy_r':mu,'profit_factor':wins/losses if losses else float('inf'),'max_drawdown_r':float(dd.min()),'sharpe':ann/sd*np.sqrt(periods_per_year) if sd else 0.,'sortino':ann/dsd*np.sqrt(periods_per_year) if dsd else 0.,'calmar':ann/abs(float(dd.min())) if dd.min()<0 else float('inf'),'tail_5pct_r':float(x.quantile(.05))}

def regime_metrics(df:pd.DataFrame, rcol='r', regime_col='regime') -> pd.DataFrame:
    rows=[]
    for regime,g in df.groupby(regime_col,dropna=False):
        m=metrics(g[rcol]); m['regime']=regime; rows.append(m)
    return pd.DataFrame(rows)

def promotion_gate(base:dict, candidate:dict, min_expectancy_delta=.05, max_drawdown_worsening=0.20) -> dict:
    delta=candidate['expectancy_r']-base['expectancy_r']; dd_base=abs(base['max_drawdown_r']); dd_cand=abs(candidate['max_drawdown_r'])
    passed=(delta>=min_expectancy_delta and candidate['profit_factor']>=base['profit_factor'] and dd_cand<=dd_base*(1+max_drawdown_worsening))
    return {'pass':bool(passed),'expectancy_delta_r':float(delta),'drawdown_ratio':float(dd_cand/max(dd_base,1e-9)),'reason':'candidate shows sufficient incremental OOS edge' if passed else 'candidate does not clear incremental OOS promotion gate'}
