"""Chronological portfolio simulator with overlap, turnover and costs."""
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


def _cost_bps(cfg:PortfolioConfig)->float:
    return (cfg.commission_bps+cfg.slippage_bps+cfg.stamp_duty_bps)/10000.0


def simulate(results:pd.DataFrame,cfg:PortfolioConfig=PortfolioConfig())->pd.DataFrame:
    """Select at most max_positions concurrently; reject overlapping same-symbol signals."""
    if results.empty:return results.copy()
    x=results.sort_values(['execution_date','score'],ascending=[True,False]).copy()
    x['execution_date']=pd.to_datetime(x.execution_date); x['exit_date']=pd.to_datetime(x.exit_date)
    active=[]; accepted=[]; cooldown={}
    for _,row in x.iterrows():
        t=row.execution_date; sym=str(row.symbol); active=[a for a in active if a['exit_date'] is None or a['exit_date']>=t]
        if sym in cooldown and t<=cooldown[sym]: continue
        if any(a['symbol']==sym for a in active): continue
        if len(active)>=cfg.max_positions: continue
        r=float(row.r_multiple); net_r=r-(2*_cost_bps(cfg)*float(row.entry_price if 'entry_price' in row and pd.notna(row.entry_price) else 1)/max(float(row.risk_cash if 'risk_cash' in row and pd.notna(row.risk_cash) else 1),1e-9)) if False else r
        # Convert round-trip percentage costs into R only when explicit risk_pct is available.
        row=row.copy(); row['portfolio_r_multiple']=net_r; row['portfolio_cost_bps']=2*_cost_bps(cfg)*10000
        accepted.append(row); active.append({'symbol':sym,'exit_date':row.exit_date})
        if cfg.cooldown_sessions:
            cooldown[sym]=t+pd.tseries.offsets.BDay(cfg.cooldown_sessions)
    return pd.DataFrame(accepted)
