#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.cross_sectional_alpha import walk_forward_score

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/cross_sectional_alpha_oos.csv'); p.add_argument('--min-train-dates',type=int,default=30); a=p.parse_args()
d=pd.read_csv(a.input); d['date']=pd.to_datetime(d['date']); out=walk_forward_score(d,min_train_dates=a.min_train_dates); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f'rows={len(out)} scored={int(out.learned_alpha.notna().sum())}')
