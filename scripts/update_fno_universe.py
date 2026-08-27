"""Refresh the current NSE stock-F&O universe.

The NSE permitted-lot-size file contains current derivative underlyings. We use
it as the authoritative gate for cash-equity short signals: a stock may only
produce a SHORT swing signal when it is currently eligible for stock F&O.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import requests

URLS=("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv","https://archives.nseindia.com/content/fo/fo_mktlots.csv")
HEADERS={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0 Safari/537.36","Accept":"text/csv,*/*"}

def main()->int:
    last=None
    for url in URLS:
        try:
            r=requests.get(url,headers=HEADERS,timeout=45); r.raise_for_status();
            if len(r.content)<500: raise ValueError("F&O file too small")
            Path("data").mkdir(exist_ok=True)
            Path("data/fno_mktlots.csv").write_bytes(r.content)
            break
        except Exception as exc:last=exc
    else: raise SystemExit(f"Unable to refresh NSE F&O universe: {last}")
    raw=pd.read_csv("data/fno_mktlots.csv")
    raw.columns=[str(c).strip() for c in raw.columns]
    col=next((c for c in raw.columns if c.upper() in {"SYMBOL","UNDERLYING","UNDERLYING SYMBOL"}),None)
    if col is None: raise SystemExit(f"No underlying symbol column in NSE F&O file: {list(raw.columns)}")
    out=pd.DataFrame({"Symbol":raw[col].astype(str).str.strip().str.upper()})
    out=out[out.Symbol.ne("") & ~out.Symbol.str.contains("NIFTY|BANKNIFTY|FINNIFTY|MIDCPNIFTY",case=False,regex=True)]
    out=out.drop_duplicates().sort_values("Symbol")
    out.to_csv("data/fno_universe.csv",index=False)
    print(f"Current stock F&O universe: {len(out)} symbols")
    return 0

if __name__=="__main__":raise SystemExit(main())
