"""Risk sizing and Monte-Carlo risk-of-ruin diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd


def position_size(capital: float, risk_pct: float, entry: float, stop: float, max_position_pct: float = .25) -> int:
    risk_cash=capital*max(0.0,risk_pct)
    per_share=abs(entry-stop)
    if per_share<=0 or entry<=0:return 0
    by_risk=int(risk_cash/per_share)
    by_exposure=int((capital*max_position_pct)/entry)
    return max(0,min(by_risk,by_exposure))


def monte_carlo_ruin(r_multiples, risk_per_trade=.01, ruin_threshold=.50, simulations=5000, trades=250, seed=42):
    r=np.asarray(r_multiples,dtype=float); r=r[np.isfinite(r)]
    if len(r)<10:return {'simulations':0,'ruin_probability':float('nan')}
    rng=np.random.default_rng(seed); ruined=0; terminal=[]
    for _ in range(simulations):
        sample=rng.choice(r,size=trades,replace=True); equity=1.; peak=1.; min_eq=1.
        for x in sample:
            equity*=max(.01,1+risk_per_trade*x); peak=max(peak,equity); min_eq=min(min_eq,equity)
            if equity<=1-ruin_threshold: ruined+=1; break
        terminal.append(equity)
    return {'simulations':simulations,'trades_per_simulation':trades,'ruin_probability':ruined/simulations,'median_terminal_equity':float(np.median(terminal)),'p05_terminal_equity':float(np.quantile(terminal,.05))}


def risk_report(results: pd.DataFrame, risk_per_trade=.01) -> dict:
    if results.empty:return {}
    return monte_carlo_ruin(results.r_multiple.to_numpy(),risk_per_trade=risk_per_trade)
