"""Signal clustering/deduplication to prevent repeated entries from one move."""
from __future__ import annotations
import pandas as pd

def deduplicate_signals(signals:pd.DataFrame, min_gap_sessions:int=3)->pd.DataFrame:
    if signals.empty:return signals.copy()
    x=signals.copy(); x['execution_date']=pd.to_datetime(x.execution_date); x=x.sort_values(['execution_date','score'],ascending=[True,False]); kept=[]; last={}
    for _,r in x.iterrows():
        key=(str(r.symbol).upper(),str(r.direction).upper())
        prev=last.get(key)
        if prev is not None and (r.execution_date-prev).days < min_gap_sessions:
            continue
        kept.append(r); last[key]=r.execution_date
    return pd.DataFrame(kept).reset_index(drop=True)
