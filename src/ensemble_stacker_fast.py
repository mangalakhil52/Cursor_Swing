"""Efficient leakage-safe stacker for fast research iterations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES=['probability','meta_probability','ensemble_probability','cross_sectional_rank']

def _features(d):
    x=d.reindex(columns=FEATURES).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    return x.fillna(x.median()).fillna(0.)

def expanding_stack_fast(df, rank_df=None, min_train=150, step=25, retrain_every=100):
    d=df.copy().reset_index(drop=True)
    if rank_df is not None:
        r=rank_df[['date','symbol','cross_sectional_rank']].copy()
        d=d.merge(r,on=['date','symbol'],how='left')
    else: d['cross_sectional_rank']=np.nan
    X=_features(d); y=pd.to_numeric(d['target_before_stop'],errors='coerce').fillna(0).astype(int)
    out=np.full(len(d),np.nan); disagreement=np.full(len(d),np.nan)
    model=None; next_refit=min_train; i=min_train
    while i<len(d):
        j=min(i+step,len(d))
        if model is None or i>=next_refit:
            if y.iloc[:i].nunique()>=2:
                model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.5,max_iter=500,class_weight='balanced'))])
                model.fit(X.iloc[:i],y.iloc[:i]); next_refit=i+retrain_every
        if model is not None:
            out[i:j]=model.predict_proba(X.iloc[i:j])[:,1]
            base=pd.to_numeric(d['ensemble_probability'].iloc[i:j],errors='coerce').fillna(.5).to_numpy()
            disagreement[i:j]=np.abs(out[i:j]-base)
        i=j
    d['stack_probability']=out; d['model_disagreement']=disagreement
    return d
