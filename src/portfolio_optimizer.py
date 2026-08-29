"""Portfolio-level selection: optimize expected edge while controlling concentration."""
from __future__ import annotations
import numpy as np
import pandas as pd


def select_diversified(candidates: pd.DataFrame, returns: pd.DataFrame, max_picks: int = 5, max_corr: float = .75, max_sector_weight: float = .40) -> pd.DataFrame:
    if candidates.empty:return candidates
    x=candidates.copy();
    if 'ml_probability' not in x:x['ml_probability']=x.get('stack_probability',x.get('probability',.5))
    if 'sector' not in x:x['sector']='UNKNOWN'
    x['risk_efficiency']=pd.to_numeric(x['ml_probability'],errors='coerce').fillna(.5)/np.maximum(pd.to_numeric(x.get('vol_annual',.2),errors='coerce').fillna(.2),.05)
    ranked=x.sort_values(['risk_efficiency','score'] if 'score' in x else ['risk_efficiency'],ascending=False).copy(); chosen=[]; sector_counts={}; sector_limit=max(1,int(np.ceil(max_picks*max_sector_weight)))
    for _,row in ranked.iterrows():
        sym=str(row.symbol); sec=str(row.sector)
        if len(chosen)>=max_picks or sector_counts.get(sec,0)>=sector_limit:continue
        if sym in returns.columns and chosen:
            corr=returns[chosen+[sym]].corr().loc[sym,chosen].abs().max()
            if pd.notna(corr) and corr>max_corr:continue
        chosen.append(sym); sector_counts[sec]=sector_counts.get(sec,0)+1
    out=ranked[ranked.symbol.astype(str).isin(chosen)].copy(); out['portfolio_selected']=True; out['portfolio_weight']=inverse_vol_weights(returns,[str(s) for s in out.symbol if str(s) in returns.columns]); return out


def inverse_vol_weights(returns: pd.DataFrame, symbols: list[str]) -> dict[str,float]:
    vols=returns[symbols].std().replace(0,np.nan).dropna()
    if vols.empty:return {}
    inv=1/vols; w=inv/inv.sum(); return {k:float(v) for k,v in w.items()}
