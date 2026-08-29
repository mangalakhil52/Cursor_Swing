"""Evaluate whether alpha adds incremental economic value to an incumbent score."""
from __future__ import annotations
import pandas as pd
import numpy as np
from .ensemble_engine import ensemble_score

def evaluate(df: pd.DataFrame, structural_col='structural_score', prob_col='probability', regime_col='regime', threshold=68., prob_floor=.60):
    rows=[]
    for mode in ('BASE','ENSEMBLE'):
        x=df.copy()
        if mode=='BASE': p=pd.Series(.60,index=x.index); score=pd.to_numeric(x[structural_col],errors='coerce')
        else:
            p=pd.to_numeric(x[prob_col],errors='coerce'); score=pd.Series([ensemble_score(s,q,r) for s,q,r in zip(x[structural_col],p,x[regime_col])],index=x.index)
        take=(score>=threshold)&(p>=prob_floor)
        y=pd.to_numeric(x['target_before_stop'],errors='coerce')
        r=np.where(y.eq(1),2.,-1.)
        r=pd.Series(r,index=x.index)[take]
        rows.append({'strategy':mode,'observations':int(take.sum()),'win_rate':float((r>0).mean()) if len(r) else 0.,'expectancy_r':float(r.mean()) if len(r) else 0.,'total_r':float(r.sum()) if len(r) else 0.})
    return pd.DataFrame(rows)
