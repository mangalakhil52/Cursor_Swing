"""Persistent incremental daily OHLCV cache shared by the scanner and research pipeline."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.data_fetcher import DataFetcher
from src.constants import IST

class CachedDataFetcher(DataFetcher):
    """Cold-start history is cached; warm runs fetch only a recent correction window."""
    def __init__(self, exchange_suffix='.NS', cache_dir='data/history', history_days=365, refresh_days=10):
        super().__init__(exchange_suffix); self.cache_dir=Path(cache_dir); self.history_days=int(history_days); self.refresh_days=int(refresh_days)
    def _path(self,symbol):
        safe=str(symbol).replace('^','INDEX_').replace('/','_').replace('\\','_'); return self.cache_dir/f'{safe}.csv'
    def _read(self,symbol):
        p=self._path(symbol)
        if not p.exists(): return pd.DataFrame()
        try:return self._normalize_single(pd.read_csv(p,index_col=0,parse_dates=True))
        except Exception:return pd.DataFrame()
    def _write(self,symbol,df):
        if df is None or df.empty:return
        self.cache_dir.mkdir(parents=True,exist_ok=True); x=df.sort_index(); x=x[~x.index.duplicated(keep='last')]; x.to_csv(self._path(symbol),date_format='%Y-%m-%dT%H:%M:%S%z')
    def fetch_daily_batch(self,symbols,days=90,chunk_size=80):
        out={}; cold=[]; warm=[]; cached={}; today=datetime.now(IST).date()
        for s in symbols:
            old=self._read(s); cached[s]=old
            if old.empty:cold.append(s); continue
            out[s]=old
            try:last=old.index[-1].date()
            except Exception:last=today
            if (today-last).days>=2:warm.append(s)
        if cold:
            fresh=super().fetch_daily_batch(cold,days=max(days,self.history_days),chunk_size=chunk_size)
            for s,df in fresh.items():self._write(s,df);out[s]=df
        if warm:
            fresh=super().fetch_daily_batch(warm,days=self.refresh_days,chunk_size=chunk_size)
            for s,df in fresh.items():
                merged=pd.concat([cached[s],df]).sort_index(); merged=merged[~merged.index.duplicated(keep='last')]; self._write(s,merged); out[s]=merged
        return {s:d for s,d in out.items() if d is not None and not d.empty}
    def fetch_daily(self,symbol,days=90):return self.fetch_daily_batch([symbol],days=days).get(symbol,pd.DataFrame())
