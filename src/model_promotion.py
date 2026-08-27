"""Objective model promotion gate. A model must beat the incumbent OOS."""
from __future__ import annotations
import pandas as pd


def evaluate_promotion(candidate: pd.DataFrame, incumbent: pd.DataFrame, min_trades=100, min_pf=1.15, min_avg_r=.05) -> dict:
    def metrics(d):
        if d.empty:return {'trades':0,'avg_r':float('nan'),'pf':0.0}
        r=pd.to_numeric(d.r_multiple,errors='coerce').dropna(); wins=r[r>0]; losses=r[r<0]
        return {'trades':len(r),'avg_r':float(r.mean()),'pf':float(wins.sum()/abs(losses.sum())) if len(losses) else float('inf')}
    c=metrics(candidate); i=metrics(incumbent)
    passed=(c['trades']>=min_trades and c['avg_r']>=min_avg_r and c['pf']>=min_pf and c['avg_r']>i['avg_r'])
    return {'candidate':c,'incumbent':i,'promote':bool(passed),'reason':'candidate passes minimum OOS thresholds and beats incumbent' if passed else 'reject: insufficient OOS evidence or no improvement over incumbent'}
