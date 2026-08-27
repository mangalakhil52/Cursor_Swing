"""Chronological portfolio simulator with overlap, turnover and realistic costs."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class PortfolioConfig:
    max_positions:int=2
    commission_bps:float=2.0
    slippage_bps:float=10.0
    stamp_duty_bps:float=1.5
    cooldown_sessions:int=0
    risk_pct:float=.01


def _round_trip_cost(cfg:PortfolioConfig)->float:
    return 2*(cfg.commission_bps+cfg.slippage_bps+cfg.stamp_duty_bps)/10000.0


def simulate(results:pd.DataFrame,cfg:PortfolioConfig=PortfolioConfig())->pd.DataFrame:
    """Take chronological non-overlapping signals, cap concurrent positions and charge round-trip costs.

    Results should contain r_multiple and exit_date. If entry_price is present, costs are
    converted to R using the configured risk percentage; otherwise gross R is retained and
    cost_bps is reported without pretending an exact R conversion is possible.
    """
    if results.empty:return results.copy()
    x=results.sort_values(['execution_date','score'],ascending=[True,False]).copy(); x['execution_date']=pd.to_datetime(x.execution_date); x['exit_date']=pd.to_datetime(x.exit_date)
    active=[]; accepted=[]; cooldown={}
    for _,row in x.iterrows():
        t=row.execution_date; sym=str(row.symbol); active=[a for a in active if a['exit_date'] is None or a['exit_date']>=t]
        if sym in cooldown and t<=cooldown[sym]:continue
        if any(a['symbol']==sym for a in active):continue
        if len(active)>=cfg.max_positions:continue
        row=row.copy(); gross=float(row.r_multiple); cost=_round_trip_cost(cfg); row['portfolio_cost_bps']=cost*10000
        if 'entry_price' in row and pd.notna(row.get('entry_price',None)):
            risk_per_share=abs(float(row.entry_price)-float(row.stop_price)) if 'stop_price' in row and pd.notna(row.get('stop_price',None)) else float(row.entry_price)*.05
            risk_pct=max(cfg.risk_pct,1e-9); cost_return=cost; row['portfolio_r_multiple']=gross-cost_return/risk_pct
        else: row['portfolio_r_multiple']=gross
        accepted.append(row); active.append({'symbol':sym,'exit_date':row.exit_date})
        if cfg.cooldown_sessions:cooldown[sym]=t+pd.tseries.offsets.BDay(cfg.cooldown_sessions)
    return pd.DataFrame(accepted)
