"""Conservative daily-bar execution/event resolver for swing research."""
from __future__ import annotations
import pandas as pd

def resolve_trade(future: pd.DataFrame, entry: float, direction: str, stop: float, target: float) -> dict:
    """Resolve the first stop/target event using daily OHLC.

    When both stop and target occur on the same daily bar, stop wins because
    intraday ordering is unknown. This is intentionally conservative.
    """
    for date, row in future.iterrows():
        hi=float(row.high); lo=float(row.low)
        if direction == 'LONG':
            hit_stop=lo <= stop; hit_target=hi >= target
            if hit_stop and hit_target: return {'exit_date':date,'exit_price':stop,'r_multiple':-1.0,'event':'STOP_AND_TARGET_SAME_BAR'}
            if hit_stop: return {'exit_date':date,'exit_price':stop,'r_multiple':-1.0,'event':'STOP'}
            if hit_target: return {'exit_date':date,'exit_price':target,'r_multiple':(target-entry)/(entry-stop),'event':'TARGET'}
        else:
            hit_stop=hi >= stop; hit_target=lo <= target
            if hit_stop and hit_target: return {'exit_date':date,'exit_price':stop,'r_multiple':-1.0,'event':'STOP_AND_TARGET_SAME_BAR'}
            if hit_stop: return {'exit_date':date,'exit_price':stop,'r_multiple':-1.0,'event':'STOP'}
            if hit_target: return {'exit_date':date,'exit_price':target,'r_multiple':(entry-target)/(stop-entry),'event':'TARGET'}
    if future.empty:return {'exit_date':None,'exit_price':entry,'r_multiple':0.0,'event':'NO_DATA'}
    close=float(future.close.iloc[-1])
    r=(close-entry)/(entry-stop) if direction=='LONG' else (entry-close)/(stop-entry)
    return {'exit_date':future.index[-1],'exit_price':close,'r_multiple':r,'event':'TIME_EXIT'}
