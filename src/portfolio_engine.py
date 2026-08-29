"""Capital-aware swing portfolio simulator.

Research approximation: each accepted trade reserves capital until its exit date, ranks
competing entries by score, caps position count/exposure, and records rejected signals.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class PortfolioConfig:
    capital: float = 1_000_000.0
    max_positions: int = 2
    max_position_pct: float = .50
    risk_pct: float = .01
    slippage_bps: float = 10.0
    fee_bps: float = 4.0

def simulate(signals: pd.DataFrame, cfg: PortfolioConfig=PortfolioConfig()) -> tuple[pd.DataFrame,pd.DataFrame]:
    if signals.empty:return signals.copy(),signals.copy()
    x=signals.copy(); x['entry_date']=pd.to_datetime(x['entry_date']); x['exit_date']=pd.to_datetime(x['exit_date']); x=x.sort_values(['entry_date','score'],ascending=[True,False])
    active=[]; accepted=[]; rejected=[]
    for _,row in x.iterrows():
        active=[a for a in active if a['exit_date']>=row.entry_date]
        reason=None
        if len(active)>=cfg.max_positions: reason='MAX_CONCURRENT_POSITIONS'
        elif any(a['symbol']==row['symbol'] for a in active): reason='SYMBOL_OVERLAP'
        else:
            entry=float(row['entry_price']); stop=float(row['stop_price'])
            risk_per_share=abs(entry-stop)
            if entry<=0 or risk_per_share<=0: reason='INVALID_ENTRY_STOP'
            else:
                risk_cash=cfg.capital*cfg.risk_pct; qty_by_risk=int(risk_cash/risk_per_share); qty_by_exposure=int(cfg.capital*cfg.max_position_pct/entry); qty=max(0,min(qty_by_risk,qty_by_exposure))
                if qty<=0: reason='ZERO_POSITION_SIZE'
                else:
                    r=float(row.get('r_multiple',0.0)); cost_return=2*(cfg.slippage_bps+cfg.fee_bps)/10000.0; net_r=r-cost_return/cfg.risk_pct
                    z=row.copy(); z['quantity']=qty; z['notional']=qty*entry; z['net_r']=net_r; z['execution_cost_bps']=2*(cfg.slippage_bps+cfg.fee_bps); accepted.append(z); active.append({'symbol':row['symbol'],'exit_date':row['exit_date']})
        if reason:
            z=row.copy(); z['rejection_reason']=reason; rejected.append(z)
    return pd.DataFrame(accepted),pd.DataFrame(rejected)
