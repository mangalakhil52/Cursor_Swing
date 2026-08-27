"""Run the point-in-time swing research engine over persistent history."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.research_engine import ResearchConfig,evaluate_symbol,summarize,decile_report,load_fno_symbols

def load_csv(path:Path)->pd.DataFrame:
    d=pd.read_csv(path); d.columns=[str(c).lower().strip() for c in d.columns]; date_col='date' if 'date' in d.columns else d.columns[0]; d[date_col]=pd.to_datetime(d[date_col],errors='coerce'); return d.dropna(subset=[date_col]).set_index(date_col).sort_index()

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out',default='reports/research.csv'); ap.add_argument('--horizon',type=int,default=10); ap.add_argument('--stop-pct',type=float,default=.05); ap.add_argument('--target-r',type=float,default=1.7); ap.add_argument('--slippage-bps',type=float,default=10); args=ap.parse_args()
    fno=load_fno_symbols(); cfg=ResearchConfig(args.horizon,args.stop_pct,args.target_r,args.slippage_bps,120,True); rows=[]
    for p in sorted(Path(args.data_dir).glob('*.csv')):
        try:
            d=load_csv(p)
            if len(d)>=cfg.min_history+cfg.horizon+2:
                r=evaluate_symbol(p.stem.upper(),d,cfg=cfg,fno_symbols=fno)
                if not r.empty:rows.append(r)
        except Exception as e:print(f'SKIP {p.name}: {e}')
    if not rows:raise SystemExit('No eligible CSV histories found')
    result=pd.concat(rows,ignore_index=True); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False); print(pd.Series(summarize(result)).to_string()); print('\nSCORE DECILES\n'); print(decile_report(result).to_string(index=False)); print(f'\nF&O short universe: {len(fno)} symbols'); print(f'Saved: {out}')

if __name__=='__main__':main()
