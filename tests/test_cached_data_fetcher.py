from pathlib import Path
import pandas as pd
from src.cached_data_fetcher import CachedDataFetcher

def bars(days=3):
    idx=pd.date_range('2026-08-24',periods=days,freq='D',tz='Asia/Kolkata')
    return pd.DataFrame({'open':[100+i for i in range(days)],'high':[102+i for i in range(days)],'low':[99+i for i in range(days)],'close':[101+i for i in range(days)],'volume':[100000+i for i in range(days)]},index=idx)

def test_cache_roundtrip(tmp_path):
    f=CachedDataFetcher(cache_dir=tmp_path,history_days=365,refresh_days=10)
    f._write('TEST',bars())
    got=f._read('TEST')
    assert len(got)==3
    assert float(got.close.iloc[-1])==103

def test_cache_path_is_safe(tmp_path):
    f=CachedDataFetcher(cache_dir=Path(tmp_path))
    assert f._path('ABC/DEF').name=='ABC_DEF.csv'
