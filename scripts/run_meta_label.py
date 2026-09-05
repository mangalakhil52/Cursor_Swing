#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.meta_label import build_features,meta_target,accept

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_ensemble_oos.csv'); p.add_argument('--output',default='reports/meta_label_oos.csv'); p.add_argument('--min-train',type=int,default=500); p.add_argument('--step',type=int,default=25); a=p.parse_args()
d=pd.read_csv(a.input).sort_values(['date','symbol']).reset_index(drop=True)
X=build_features(d); y=meta_target(d); preds=np.full(len(d),np.nan)
for i in range(a.min_train,len(d),a.step):
    j=min(i+a.step,len(d))
    if y.iloc[:i].nunique()<2: continue
    med=X.iloc[:i].median().fillna(0.)
    train=X.iloc[:i].fillna(med).fillna(0.); test=X.iloc[i:j].fillna(med).fillna(0.)
    model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=1000,class_weight='balanced',C=.5))])
    model.fit(train,y.iloc[:i]); preds[i:j]=model.predict_proba(test)[:,1]
d['meta_probability']=preds; d['meta_accept']=accept(d,d.meta_probability); Path(a.output).parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.output,index=False); print(f'rows={len(d)} scored={int(pd.Series(preds).notna().sum())} accepted={int(d.meta_accept.sum())}')
