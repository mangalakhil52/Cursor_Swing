"""Stress-test strategy edge under adverse regime and cost assumptions."""
from __future__ import annotations
import numpy as np
import pandas as pd

def stress_grid(df: pd.DataFrame, cost_r=(0,.05,.10,.20), win_haircuts=(0,.05,.10), loss_expansions=(0,.10,.20)) -> pd.DataFrame:
    rows=[]
    for cost,hair,loss_exp in __import__('itertools').product(cost_r,win_haircuts,loss_expansions):
        y=pd.to_numeric(df['target_before_stop'],errors='coerce').dropna().astype(int)
        r=np.where(y.eq(1),2*(1-hair)-cost,-1*(1+loss_exp)-cost)
        rows.append({'cost_r':cost,'win_haircut':hair,'loss_expansion':loss_exp,'trades':len(r),'expectancy_r':float(np.mean(r)) if len(r) else np.nan,'win_rate':float(np.mean(r>0)) if len(r) else np.nan})
    return pd.DataFrame(rows)

def adversarial_score(grid:pd.DataFrame)->dict:
    x=grid.dropna(subset=['expectancy_r'])
    if x.empty:return {'robust':False,'worst_expectancy_r':np.nan,'positive_stress_share':0.}
    return {'robust':bool(x.expectancy_r.min()>0),'worst_expectancy_r':float(x.expectancy_r.min()),'positive_stress_share':float((x.expectancy_r>0).mean())}
