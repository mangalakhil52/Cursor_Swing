"""Conservative covariance-aware allocation for selected swing positions."""
from __future__ import annotations
import numpy as np
import pandas as pd

def risk_parity_weights(returns:pd.DataFrame, symbols:list[str], max_weight=.35, min_weight=.05)->dict[str,float]:
    syms=[s for s in symbols if s in returns.columns]
    if not syms:return {}
    cov=returns[syms].cov().to_numpy(dtype=float); cov=np.nan_to_num(cov); cov+=np.eye(len(syms))*1e-8
    vol=np.sqrt(np.diag(cov)); inv=1/np.maximum(vol,1e-8); w=inv/inv.sum()
    w=np.minimum(w,max_weight); w=w/w.sum(); w=np.maximum(w,min_weight); w=w/w.sum()
    return {s:float(v) for s,v in zip(syms,w)}

def portfolio_stats(returns:pd.DataFrame,weights:dict[str,float])->dict:
    syms=list(weights); w=np.array([weights[s] for s in syms]); cov=returns[syms].cov().fillna(0).to_numpy();
    return {'annualized_vol':float(np.sqrt(max(w@cov@w,0))*np.sqrt(252)),'effective_positions':float(1/(w@w))}
