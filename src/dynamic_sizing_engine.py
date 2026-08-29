"""Adaptive position sizing combining edge, volatility, correlation and drawdown state."""
from __future__ import annotations
import math

def size_position(*,capital,entry,stop,probability,avg_win_r=2.,avg_loss_r=1.,vol_annual=.20,target_vol=.20,drawdown_pct=0.,correlation_penalty=1.,base_risk=.01,max_risk=.015,max_exposure=.50,kelly_fraction=.25):
    entry=float(entry); stop=float(stop); p=max(0.,min(1.,float(probability))); b=max(avg_win_r/max(avg_loss_r,1e-9),1e-9)
    k=max(0.,(b*p-(1-p))/b)*kelly_fraction
    vol_mult=min(1.,target_vol/max(float(vol_annual),1e-9))
    dd_mult=1. if drawdown_pct<5 else .75 if drawdown_pct<8 else .50 if drawdown_pct<12 else 0.
    risk_pct=min(max_risk,max(base_risk,k))*vol_mult*dd_mult*max(0.,min(1.,correlation_penalty))
    distance=abs(entry-stop)/max(entry,1e-9); risk_cash=capital*risk_pct
    qty=math.floor(risk_cash/max(entry*distance,1e-9)); qty=min(qty,math.floor(capital*max_exposure/max(entry,1e-9)))
    return {'quantity':max(0,qty),'risk_pct':risk_pct,'risk_cash':capital*risk_pct,'notional':max(0,qty)*entry,'kelly_fraction':k,'volatility_multiplier':vol_mult,'drawdown_multiplier':dd_mult,'correlation_multiplier':correlation_penalty}
