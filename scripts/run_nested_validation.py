#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.nested_validation import make_nested_folds, choose_threshold

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/nested_validation.csv'); a=p.parse_args()
d=pd.read_csv(a.input).sort_values(['date','symbol']).reset_index(drop=True)
rows=[]
for i,f in enumerate(make_nested_folds(len(d)),1):
    tr=d.iloc[f.train_start:f.train_end]; va=d.iloc[f.validation_start:f.validation_end]; te=d.iloc[f.test_start:f.test_end]
    choice=choose_threshold(tr,va)
    if choice is None: continue
    x=te[te.probability>=choice['threshold']]
    rows.append({'fold':i,'threshold_selected':choice['threshold'],'validation_expectancy_r':choice['validation_expectancy_r'],'validation_trades':choice['validation_trades'],'test_trades':len(x),'test_win_rate':float(x.target_before_stop.mean()) if len(x) else 0.0,'test_expectancy_r':float((2*x.target_before_stop-(1-x.target_before_stop)).mean()) if len(x) else 0.0})
out=pd.DataFrame(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))
