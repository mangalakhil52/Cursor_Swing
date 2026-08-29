"""Portfolio exposure controls for correlated swing positions."""
from __future__ import annotations
import numpy as np
import pandas as pd

def correlation_matrix(returns:pd.DataFrame, lookback:int=60)->pd.DataFrame:
    return returns.tail(lookback).pct_change().corr(min_periods=max(20,lookback//3))

def beta_to_market(stock_returns:pd.Series, market_returns:pd.Series, lookback:int=60)->float:
    x=pd.concat([stock_returns.tail(lookback),market_returns.tail(lookback)],axis=1).dropna()
    if len(x)<20:return float('nan')
    return float(x.iloc[:,0].cov(x.iloc[:,1])/max(x.iloc[:,1].var(ddof=1),1e-12))

def approve(candidate:dict, active:list[dict], corr:pd.DataFrame|None=None, *, max_sector=1, max_abs_beta=1.5, max_corr=.75)->tuple[bool,str]:
    sector=candidate.get('sector'); beta=float(candidate.get('beta',0.0)); symbol=candidate.get('symbol')
    if sector and sum(a.get('sector')==sector for a in active)>=max_sector:return False,'SECTOR_CONCENTRATION'
    if abs(beta)>max_abs_beta:return False,'MARKET_BETA_LIMIT'
    if corr is not None and symbol in corr.index:
        for a in active:
            other=a.get('symbol')
            if other in corr.columns and pd.notna(corr.loc[symbol,other]) and abs(float(corr.loc[symbol,other]))>max_corr:return False,'CORRELATION_CLUSTER'
    return True,'APPROVED'
