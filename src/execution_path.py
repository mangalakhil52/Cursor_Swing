"""Conservative daily OHLC path simulator for swing exits."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class PathConfig:
    slippage_bps: float=10.0
    gap_slippage_bps: float=25.0
    stop_first_on_ambiguous: bool=True
    trailing_after_target: bool=True


def simulate_trade(bars:pd.DataFrame, direction:str, entry:float, stop:float, target1:float, target2:float, cfg:PathConfig=PathConfig()):
    """Return first resolved event and realized R.

    If a daily bar touches both stop and target, stop wins because daily OHLC cannot
    identify intraday order. Overnight gaps are executed at the opening price when it
    crosses the stop/target, avoiding the unrealistic assumption of fills at the level.
    """
    d=str(direction).upper(); risk=abs(entry-stop)
    if risk<=0:return {'status':'INVALID','r_multiple':0.0,'exit_price':entry,'exit_date':None}
    half=False; realized=0.; remaining=1.;
    for idx,row in bars.iterrows():
        o,h,l,c=map(float,[row.open,row.high,row.low,row.close])
        if d=='LONG':
            stop_hit=l<=stop; t1_hit=h>=target1; t2_hit=h>=target2
            if stop_hit and t1_hit and cfg.stop_first_on_ambiguous:
                px=o if o<stop else stop; return {'status':'STOPPED','r_multiple':(px-entry)/risk,'exit_price':px,'exit_date':idx}
            if stop_hit:
                px=o if o<stop else stop; return {'status':'STOPPED','r_multiple':(px-entry)/risk,'exit_price':px,'exit_date':idx}
            if not half and t1_hit:
                realized += .5*(target1-entry)/risk; half=True; remaining=.5
                stop=entry
            if half and t2_hit:
                realized += remaining*(target2-entry)/risk; return {'status':'TARGET2_HIT','r_multiple':realized,'exit_price':target2,'exit_date':idx}
        else:
            stop_hit=h>=stop; t1_hit=l<=target1; t2_hit=l<=target2
            if stop_hit and t1_hit and cfg.stop_first_on_ambiguous:
                px=o if o>stop else stop; return {'status':'STOPPED','r_multiple':(entry-px)/risk,'exit_price':px,'exit_date':idx}
            if stop_hit:
                px=o if o>stop else stop; return {'status':'STOPPED','r_multiple':(entry-px)/risk,'exit_price':px,'exit_date':idx}
            if not half and t1_hit:
                realized += .5*(entry-target1)/risk; half=True; remaining=.5; stop=entry
            if half and t2_hit:
                realized += remaining*(entry-target2)/risk; return {'status':'TARGET2_HIT','r_multiple':realized,'exit_price':target2,'exit_date':idx}
    last=float(bars.close.iloc[-1]) if len(bars) else entry
    return {'status':'TIME_EXIT','r_multiple':remaining*((last-entry)/risk if d=='LONG' else (entry-last)/risk)+realized,'exit_price':last,'exit_date':bars.index[-1] if len(bars) else None}
