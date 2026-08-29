"""Portfolio-level stress simulation for swing-trading research.

Uses empirical block bootstrap so clustered wins/losses are retained better than an IID
shuffle. Correlation can be supplied as a position-return correlation matrix.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def max_drawdown(x):
    eq=np.cumsum(x); return float((eq-np.maximum.accumulate(eq)).min())

def monte_carlo_risk(r, simulations=5000, horizon=100, block=5, seed=42, ruin_r=-10.0):
    x=np.asarray(r,dtype=float); x=x[np.isfinite(x)]
    if len(x)<10:return {'simulations':0,'ruin_probability':float('nan'),'median_terminal_r':float('nan'),'p05_terminal_r':float('nan'),'p95_max_drawdown_r':float('nan')}
    rng=np.random.default_rng(seed); terminals=[]; dds=[]; ruins=0
    blocks=max(1,min(int(block),len(x))); nblocks=int(np.ceil(horizon/blocks))
    for _ in range(simulations):
        path=[]
        for _ in range(nblocks):
            i=int(rng.integers(0,len(x)-blocks+1)); path.extend(x[i:i+blocks])
        path=np.asarray(path[:horizon]); terminals.append(path.sum()); dds.append(max_drawdown(path))
        ruins += int(np.min(np.cumsum(path))<=ruin_r)
    return {'simulations':simulations,'ruin_probability':float(ruins/simulations),'median_terminal_r':float(np.median(terminals)),'p05_terminal_r':float(np.quantile(terminals,.05)),'p95_terminal_r':float(np.quantile(terminals,.95)),'median_max_drawdown_r':float(np.median(dds)),'p95_max_drawdown_r':float(np.quantile(dds,.05))}

def portfolio_day_returns(position_returns:pd.DataFrame, weights=None):
    x=position_returns.copy().sort_index().fillna(0.)
    if weights is None: weights=np.ones(x.shape[1])/max(x.shape[1],1)
    w=np.asarray(weights,dtype=float); w=w/max(w.sum(),1e-12)
    return x.to_numpy()@w
