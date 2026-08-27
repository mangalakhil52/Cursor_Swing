"""Market-regime classification used for conditional research."""
from __future__ import annotations
import numpy as np
import pandas as pd


def classify_market(benchmark: pd.DataFrame) -> str:
    c=benchmark.close.astype(float); r=c.pct_change();
    ema21=c.ewm(span=21,adjust=False).mean().iloc[-1]; ema50=c.ewm(span=50,adjust=False).mean().iloc[-1]
    vol=float(r.rolling(20).std().iloc[-1]*np.sqrt(252)); ret20=float(c.pct_change(20).iloc[-1])
    if c.iloc[-1]>ema21>ema50 and ret20>0.03:return "STRONG_BULL"
    if c.iloc[-1]>ema50 and ret20>0:return "BULL"
    if c.iloc[-1]<ema21<ema50 and ret20<-0.03:return "STRONG_BEAR"
    if c.iloc[-1]<ema50 and ret20<0:return "BEAR"
    return "HIGH_VOL_SIDEWAYS" if vol>.35 else "SIDEWAYS"
