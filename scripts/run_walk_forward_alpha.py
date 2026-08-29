#!/usr/bin/env python3
"""Evaluate a simple, leakage-resistant alpha baseline across chronological folds.

This is deliberately a benchmark model, not a production predictor. It establishes whether
point-in-time features contain incremental information before introducing more complex ML.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score
from src.walk_forward import make_folds

FEATURES=['ret_3','ret_5','ret_10','ret_20','ret_60','vol_20','atr_pct','efficiency_20','ema9_gap','ema21_gap','ema_spread','volume_ratio','range_position_20','residual_20','path_up_fraction_20','path_down_fraction_20']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='reports/alpha_dataset.parquet'); ap.add_argument('--output',default='reports/alpha_oos_predictions.csv'); ap.add_argument('--train-size',type=int,default=10000); ap.add_argument('--test-size',type=int,default=2000); ap.add_argument('--step',type=int,default=2000); ap.add_argument('--embargo',type=int,default=10); args=ap.parse_args()
    d=pd.read_parquet(args.input).sort_values(['date','symbol']).reset_index(drop=True)
    d['y']=pd.to_numeric(d['target_before_stop'],errors='coerce'); d=d.dropna(subset=['y'])
    rows=[]
    for fold_no,f in enumerate(make_folds(len(d),args.train_size,args.test_size,args.step,args.embargo),1):
        tr=d.iloc[f.train_start:f.train_end]; te=d.iloc[f.test_start:f.test_end]
        if tr.y.nunique()<2: continue
        model=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=200,l2_regularization=1.0,random_state=42))
        model.fit(tr[FEATURES],tr.y.astype(int)); p=model.predict_proba(te[FEATURES])[:,1]
        z=te[['date','symbol','target_before_stop','forward_return_5','forward_return_10']].copy(); z['probability']=p; z['fold']=fold_no; rows.append(z)
    if not rows: raise SystemExit('No valid walk-forward folds')
    out=pd.concat(rows,ignore_index=True); Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
    y=out.target_before_stop.astype(int); p=out.probability
    summary={'folds':int(out.fold.nunique()),'oos_observations':len(out),'brier':float(brier_score_loss(y,p)),'roc_auc':float(roc_auc_score(y,p)) if y.nunique()>1 else float('nan'),'mean_probability':float(p.mean()),'actual_win_rate':float(y.mean())}
    pd.Series(summary).to_csv(Path(args.output).with_name('alpha_oos_summary.csv'),header=['value']); print(pd.Series(summary).to_string())
if __name__=='__main__': main()
