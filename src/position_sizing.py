"""Conservative dynamic position sizing for swing research."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SizingConfig:
    capital: float=1_000_000.0
    base_risk_pct: float=.01
    max_position_pct: float=.50
    kelly_fraction: float=.25
    max_kelly_risk_pct: float=.015
    target_vol_annual: float=.20
    min_risk_pct: float=.0025
    max_risk_pct: float=.015

def fractional_kelly(p:float, avg_win_r:float=2., avg_loss_r=1., fraction:float=.25)->float:
    p=max(0.,min(1.,float(p))); q=1-p; b=max(float(avg_win_r)/max(avg_loss_r,1e-9),1e-9)
    return max(0.,min(1.,fraction*((b*p-q)/b)))

def size(entry:float, stop:float, probability:float, realized_vol_annual:float|None=None, drawdown_pct:float=0., cfg:SizingConfig=SizingConfig())->dict:
    entry=float(entry); stop=float(stop); p=float(probability); risk_dist=abs(entry-stop)/max(entry,1e-12)
    k=fractional_kelly(p,fraction=cfg.kelly_fraction); risk_pct=min(cfg.max_kelly_risk_pct,max(cfg.min_risk_pct,max(cfg.base_risk_pct,k)))
    if realized_vol_annual and realized_vol_annual>0:
        vol_mult=min(1.0,cfg.target_vol_annual/float(realized_vol_annual)); risk_pct*=vol_mult
    if drawdown_pct>=12: risk_pct=0.
    elif drawdown_pct>=8: risk_pct*=.5
    risk_pct=max(0.,min(cfg.max_risk_pct,risk_pct)); risk_cash=cfg.capital*risk_pct
    qty_by_risk=math.floor(risk_cash/max(entry*risk_dist,1e-9)); qty_by_exposure=math.floor(cfg.capital*cfg.max_position_pct/max(entry,1e-9)); qty=max(0,min(qty_by_risk,qty_by_exposure))
    return {'risk_pct':risk_pct,'risk_cash':risk_cash,'quantity':qty,'notional':qty*entry,'stop_distance_pct':risk_dist,'kelly_risk_pct':k,'vol_multiplier':(min(1.,cfg.target_vol_annual/realized_vol_annual) if realized_vol_annual and realized_vol_annual>0 else 1.)}
