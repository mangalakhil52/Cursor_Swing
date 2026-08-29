#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.meta_label import build_features,meta_target,accept

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/meta_label_oos.csv'); a=p.parse_args()
d=pd.read_csv(a.input).sort_values(['date','symbol']).reset_index(drop=True)
X=build_features(d); y=meta_target(d); preds=np.full(len(d),np.nan)
# Strict expanding walk-forward: model only sees earlier observations.
for i in range(max(100,len(d)//3),len(d),max(20,len(d)//20)):
    train=slice(0,i); test=slice(i,min(i+max(20,len(d)//20),len(d)))
    if y.iloc[train].nunique()<2: continue
    model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=1000,class_weight='balanced'))]); model.fit(X.iloc[train],y.iloc[train]); preds[test]=model.predict_proba(X.iloc[test])[:,1]
d['meta_probability']=preds; d['meta_accept']=accept(d,d.meta_probability); Path(a.output).parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.output,index=False); print(f'rows={len(d)} accepted={int(d.meta_accept.sum())}')
