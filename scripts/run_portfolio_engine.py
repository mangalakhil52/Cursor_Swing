#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.portfolio_engine import simulate,PortfolioConfig

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/portfolio_candidates.csv'); p.add_argument('--output',default='reports/portfolio_selected.csv'); p.add_argument('--capital',type=float,default=1000000); p.add_argument('--max-positions',type=int,default=2); a=p.parse_args()
d=pd.read_csv(a.input); accepted,rejected=simulate(d,PortfolioConfig(capital=a.capital,max_positions=a.max_positions))
Path(a.output).parent.mkdir(parents=True,exist_ok=True); accepted.to_csv(a.output,index=False); rejected.to_csv(Path(a.output).with_name('portfolio_rejected.csv'),index=False)
print(f'accepted={len(accepted)} rejected={len(rejected)}')
