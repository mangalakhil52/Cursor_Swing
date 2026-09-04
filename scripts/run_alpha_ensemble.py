#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.alpha_ensemble import expanding_ensemble

p=argparse.ArgumentParser()
p.add_argument('--input',default='reports/alpha_oos_predictions.csv')
p.add_argument('--output',default='reports/alpha_ensemble_oos.csv')
p.add_argument('--min-train',type=int,default=150)
p.add_argument('--step',type=int,default=25)
p.add_argument('--retrain-every',type=int,default=100)
a=p.parse_args()
d=pd.read_csv(a.input).sort_values(['date','symbol']).reset_index(drop=True)
out=expanding_ensemble(d,min_train=a.min_train,step=a.step,retrain_every=a.retrain_every)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.output,index=False)
print(f'rows={len(out)} scored={out.ensemble_probability.notna().sum()} retrain_every={a.retrain_every}')
