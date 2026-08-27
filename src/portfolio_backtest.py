"""Portfolio-level backtest controls for overlapping signals and exposure caps."""
from __future__ import annotations
import pandas as pd


def remove_overlapping_signals(results: pd.DataFrame, max_concurrent: int = 2) -> pd.DataFrame:
    """Keep only signals that could coexist under a simple max-concurrent model.

    Signals are ordered by execution date then score. A new signal is accepted
    only when fewer than `max_concurrent` accepted trades have an exit date
    strictly after its execution date. This is a conservative research filter;
    it does not claim to model broker fills or intraday portfolio rebalancing.
    """
    if results.empty:return results.copy()
    x=results.copy(); x['execution_date']=pd.to_datetime(x['execution_date']); x['exit_date']=pd.to_datetime(x['exit_date'])
    x=x.sort_values(['execution_date','score'],ascending=[True,False]).reset_index(drop=True)
    accepted=[]
    for _,row in x.iterrows():
        active=[a for a in accepted if pd.notna(a['exit_date']) and a['exit_date']>row.execution_date]
        if len(active)<max_concurrent: accepted.append(row.to_dict())
    return pd.DataFrame(accepted,columns=x.columns)


def exposure_report(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:return pd.DataFrame()
    x=results.copy(); x['execution_date']=pd.to_datetime(x['execution_date']); x['exit_date']=pd.to_datetime(x['exit_date'])
    dates=sorted(x.execution_date.dropna().unique()); rows=[]
    for d in dates:
        active=x[(x.execution_date<=d)&(x.exit_date>=d)]
        rows.append({'date':d,'active_trades':len(active),'long_trades':int((active.direction=='LONG').sum()),'short_trades':int((active.direction=='SHORT').sum())})
    return pd.DataFrame(rows)
