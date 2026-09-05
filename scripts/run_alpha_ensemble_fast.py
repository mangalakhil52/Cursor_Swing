#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.alpha_ensemble import expanding_ensemble

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--retrain-every',type=int,default=100)
a=p.parse_args()
d=pd.read_csv(a.input).sort_values(['date','symbol']).reset_index(drop=True)
out=expanding_ensemble(d,retrain_every=a.retrain_every)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.output,index=False)
print(f'rows={len(out)} scored={out.ensemble_probability.notna().sum()} retrain_every={a.retrain_every}')
