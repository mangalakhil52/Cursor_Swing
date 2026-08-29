#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.execution_portfolio import simulate_candidates

p=argparse.ArgumentParser(); p.add_argument('--candidates',default='reports/portfolio_candidates.csv'); p.add_argument('--history',default='data/history_daily'); p.add_argument('--output',default='reports/execution_portfolio.csv'); p.add_argument('--capital',type=float,default=1_000_000); p.add_argument('--max-positions',type=int,default=2); a=p.parse_args()
c=pd.read_csv(a.candidates); bars={}
for sym in c.symbol.astype(str).unique():
    f=Path(a.history)/(sym+'.csv')
    if f.exists(): bars[sym]=pd.read_csv(f,index_col=0,parse_dates=True).sort_index()
accepted,rejected=simulate_candidates(c,bars,a.capital,a.max_positions)
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); accepted.to_csv(out,index=False); rejected.to_csv(out.with_name('execution_portfolio_rejected.csv'),index=False)
print(f'accepted={len(accepted)} rejected={len(rejected)}')
