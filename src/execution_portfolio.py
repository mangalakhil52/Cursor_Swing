"""Portfolio simulation using actual OHLC exit paths and execution costs."""
from __future__ import annotations
import pandas as pd
from .execution_path import simulate_trade, PathConfig
from .position_sizing import size
from .execution_costs import estimate_cost_bps


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
        future_bars=bars.loc[bars.index>entry_date]
        s=size(entry=float(c.entry_price),stop=float(c.stop_price),probability=float(c.get('probability',.6)),realized_vol_annual=float(c.get('vol_annual',.2)),capital=capital)
        if s['quantity']<=0:
            z=c.to_dict(); z['rejection_reason']='ZERO_POSITION_SIZE'; rejected.append(z); continue
        if future_bars.empty:
            z=c.to_dict(); z['rejection_reason']='NO_POST_ENTRY_BARS'; rejected.append(z); continue
        path=simulate_trade(future_bars,str(c.direction),float(c.entry_price),float(c.stop_price),float(c.target1),float(c.target2),PathConfig())
        cost=estimate_cost_bps(price=float(c.entry_price),quantity=float(s['quantity']),adv_value=float(c.get('adv_value',0.) or 0.))
        risk_dist=abs(float(c.entry_price)-float(c.stop_price))/max(float(c.entry_price),1e-12)
        cost_r=(cost['total_cost_bps']/10000.)/max(risk_dist,1e-12)
        z=c.to_dict(); z.update(s); z.update(path)
        z.update({'execution_cost_bps':float(cost['total_cost_bps']),'participation':float(cost['participation']),'net_r':float(path['r_multiple']-cost_r),'gross_r':float(path['r_multiple'])})
        z['pnl_cash']=float(z['net_r'])*capital*s['risk_pct']
        accepted.append(z)
        if path.get('exit_date') is not None:
            active.append({'symbol':sym,'exit_date':pd.to_datetime(path['exit_date'])})
    return pd.DataFrame(accepted),pd.DataFrame(rejected)
