#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.cross_sectional_rank import rank_by_date

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/cross_sectional_rank.csv'); p.add_argument('--top-k',type=int,default=10); a=p.parse_args()
d=pd.read_csv(a.input); d['date']=pd.to_datetime(d['date']); out=rank_by_date(d,top_k=a.top_k); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f'rows={len(out)} selected={int(out.selected.sum())}')
