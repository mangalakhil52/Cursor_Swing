"""Portfolio-level selection: avoid concentrated, highly correlated bets."""
from __future__ import annotations
import numpy as np
import pandas as pd


def select_diversified(candidates: pd.DataFrame, returns: pd.DataFrame, max_picks: int = 2, max_corr: float = .75) -> pd.DataFrame:
    if candidates.empty:return candidates
    ranked=candidates.sort_values(["ml_probability","score"],ascending=False).copy()
    chosen=[]
    for _,row in ranked.iterrows():
        sym=str(row.symbol)
        if sym not in returns.columns:
            if not chosen: chosen.append(sym)
            elif len(chosen)<max_picks: chosen.append(sym)
        else:
            if not chosen: chosen.append(sym)
            elif len(chosen)<max_picks:
                corr=returns[chosen+[sym]].corr().loc[sym,chosen].abs().max()
                if pd.isna(corr) or corr<=max_corr: chosen.append(sym)
        if len(chosen)>=max_picks:break
    return ranked[ranked.symbol.astype(str).isin(chosen)].copy()


def inverse_vol_weights(returns: pd.DataFrame, symbols: list[str]) -> dict[str,float]:
    vols=returns[symbols].std().replace(0,np.nan).dropna()
    if vols.empty:return {}
    inv=1/vols; w=inv/inv.sum(); return {k:float(v) for k,v in w.items()}
