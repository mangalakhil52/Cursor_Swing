#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.alpha_ensemble import expanding_ensemble

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/alpha_ensemble_oos.csv'); a=p.parse_args()
d=pd.read_csv(a.input).sort_values(['date','symbol']).reset_index(drop=True); out=expanding_ensemble(d); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f'rows={len(out)} scored={out.ensemble_probability.notna().sum()}')
