#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.ensemble_stacker_fast import expanding_stack_fast

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/meta_label_oos.csv'); p.add_argument('--output',default='reports/ensemble_stack_oos.csv'); p.add_argument('--rank-input',default='reports/cross_sectional_rank.csv'); p.add_argument('--retrain-every',type=int,default=100); p.add_argument('--step',type=int,default=25); p.add_argument('--min-train',type=int,default=150); a=p.parse_args()
d=pd.read_csv(a.input).sort_values(['date','symbol']).reset_index(drop=True); r=pd.read_csv(a.rank_input).sort_values(['date','symbol']).reset_index(drop=True)
out=expanding_stack_fast(d,r,min_train=a.min_train,step=a.step,retrain_every=a.retrain_every)
Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
print(f'rows={len(out)} scored={out.stack_probability.notna().sum()} retrain_every={a.retrain_every}')
