"""Portfolio simulation using actual OHLC exit paths instead of fixed R labels."""
from __future__ import annotations
import pandas as pd
from .execution_path import simulate_trade, PathConfig
from .position_sizing import size


def simulate_candidates(candidates: pd.DataFrame, bars_by_symbol: dict[str,pd.DataFrame], capital: float=1_000_000., max_positions: int=2):
    active=[]; accepted=[]; rejected=[]
    for _,c in candidates.sort_values(['entry_date','score'],ascending=[True,False]).iterrows():
        entry_date=pd.to_datetime(c.entry_date)
        active=[a for a in active if a['exit_date']>=entry_date]
        if len(active)>=max_positions:
            z=c.to_dict(); z['rejection_reason']='MAX_CONCURRENT_POSITIONS'; rejected.append(z); continue
        sym=str(c.symbol); bars=bars_by_symbol.get(sym)
        if bars is None or bars.empty:
            z=c.to_dict(); z['rejection_reason']='MISSING_EXECUTION_BARS'; rejected.append(z); continue
        bars=bars.copy(); bars.index=pd.to_datetime(bars.index)
        # Entry is the signal-date close. Only bars strictly after the signal date may
        # determine exits; otherwise the simulator would use pre-entry intraday extremes.
        future_bars=bars.loc[bars.index>entry_date]
        s=size(entry=float(c.entry_price),stop=float(c.stop_price),probability=float(c.get('probability',.6)),realized_vol_annual=float(c.get('vol_annual',.2)),capital=capital)
        if s['quantity']<=0:
            z=c.to_dict(); z['rejection_reason']='ZERO_POSITION_SIZE'; rejected.append(z); continue
        if future_bars.empty:
            z=c.to_dict(); z['rejection_reason']='NO_POST_ENTRY_BARS'; rejected.append(z); continue
        path=simulate_trade(future_bars,str(c.direction),float(c.entry_price),float(c.stop_price),float(c.target1),float(c.target2),PathConfig())
        z=c.to_dict(); z.update(s); z.update(path)
        z['pnl_cash']=float(z['net_r'] if 'net_r' in z else path['r_multiple'])*capital*s['risk_pct']
        accepted.append(z)
        if path.get('exit_date') is not None:
            active.append({'symbol':sym,'exit_date':pd.to_datetime(path['exit_date'])})
    return pd.DataFrame(accepted),pd.DataFrame(rejected)
