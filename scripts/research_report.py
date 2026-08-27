"""Run point-in-time research with persistent Nifty regime/benchmark data."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.research_engine import ResearchConfig,evaluate_symbol,summarize,decile_report,load_fno_symbols
from src.robustness import symbol_concentration,block_stability
from src.regime_engine import classify_market
from src.portfolio_backtest import remove_overlapping_signals,exposure_report

def load_csv(path:Path)->pd.DataFrame:
    d=pd.read_csv(path); d.columns=[str(c).lower().strip() for c in d.columns]; date_col='date' if 'date' in d.columns else d.columns[0]; d[date_col]=pd.to_datetime(d[date_col],errors='coerce'); return d.dropna(subset=[date_col]).set_index(date_col).sort_index()

def regime_at(benchmark:pd.DataFrame,date)->str:
    hist=benchmark.loc[benchmark.index<=pd.Timestamp(date)]
    if len(hist)<60:return 'INSUFFICIENT_HISTORY'
    return classify_market(hist)

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--benchmark',default='data/history/_benchmark.csv'); ap.add_argument('--out',default='reports/research.csv'); ap.add_argument('--horizon',type=int,default=10); ap.add_argument('--stop-pct',type=float,default=.05); ap.add_argument('--target-r',type=float,default=1.7); ap.add_argument('--slippage-bps',type=float,default=10); ap.add_argument('--max-concurrent',type=int,default=2); args=ap.parse_args()
    fno=load_fno_symbols(); benchmark=load_csv(Path(args.benchmark)) if Path(args.benchmark).exists() else None; cfg=ResearchConfig(args.horizon,args.stop_pct,args.target_r,args.slippage_bps,120,True); rows=[]
    for p in sorted(Path(args.data_dir).glob('*.csv')):
        if p.name.startswith('_'):continue
        try:
            d=load_csv(p)
            if len(d)>=cfg.min_history+cfg.horizon+2:
                b=benchmark.close if benchmark is not None and 'close' in benchmark.columns else None; r=evaluate_symbol(p.stem.upper(),d,benchmark=b,cfg=cfg,fno_symbols=fno)
                if not r.empty:rows.append(r)
        except Exception as e:print(f'SKIP {p.name}: {e}')
    if not rows:raise SystemExit('No eligible CSV histories found')
    result=pd.concat(rows,ignore_index=True)
    if benchmark is not None:result['regime']=result['signal_date'].map(lambda x:regime_at(benchmark,x))
    else:result['regime']='NO_BENCHMARK'
    portfolio=remove_overlapping_signals(result,args.max_concurrent)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False); portfolio.to_csv(out.with_name('portfolio_backtest.csv'),index=False)
    pd.Series(summarize(result)).to_csv(out.with_name('summary.csv'),header=['value']); pd.Series(summarize(portfolio)).to_csv(out.with_name('portfolio_summary.csv'),header=['value'])
    decile_report(result).to_csv(out.with_name('score_deciles.csv'),index=False); symbol_concentration(result).to_csv(out.with_name('symbol_concentration.csv'),index=False); block_stability(result).to_csv(out.with_name('time_blocks.csv'),index=False); exposure_report(portfolio).to_csv(out.with_name('portfolio_exposure.csv'),index=False)
    result.groupby(['regime','direction'],observed=True).agg(trades=('r_multiple','size'),win_rate=('r_multiple',lambda s:(s>0).mean()),avg_r=('r_multiple','mean'),cum_r=('r_multiple','sum')).reset_index().to_csv(out.with_name('regime_direction.csv'),index=False)
    print('ALL SIGNALS'); print(pd.Series(summarize(result)).to_string()); print('\nPORTFOLIO-CAPPED'); print(pd.Series(summarize(portfolio)).to_string()); print(f'\nF&O short universe: {len(fno)} symbols'); print(f'Saved: {out}')

if __name__=='__main__':main()
