"""Point-in-time cross-sectional ranking of eligible swing candidates."""
from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS={'probability':.30,'structural_score':.20,'residual_momentum':.15,'relative_strength':.15,'volatility_efficiency':.10,'regime_fit':.10}

def _pct_rank(s): return s.rank(pct=True,method='average')

def rank_candidates(df:pd.DataFrame, weights=None, top_k=10)->pd.DataFrame:
    w=weights or DEFAULT_WEIGHTS; x=df.copy()
    for c in w:
        if c not in x:x[c]=0.
        x[c]=pd.to_numeric(x[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        x[c]=x[c].fillna(x[c].median() if x[c].notna().any() else 0.)
    for c in w:x[c+'_rank']=_pct_rank(x[c])
    x['cross_sectional_score']=sum(float(v)*x[c+'_rank'] for c,v in w.items())
    x=x.sort_values('cross_sectional_score',ascending=False).reset_index(drop=True)
    x['cross_sectional_rank']=np.arange(1,len(x)+1); x['selected']=x.cross_sectional_rank<=top_k
    return x

def rank_by_date(df:pd.DataFrame,date_col='date',top_k=10)->pd.DataFrame:
    return pd.concat([rank_candidates(g,top_k=top_k) for _,g in df.groupby(date_col,sort=True)],ignore_index=True) if len(df) else df.copy()
