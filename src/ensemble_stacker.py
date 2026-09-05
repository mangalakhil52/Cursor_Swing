"""Stack heterogeneous OOS model predictions into a calibrated meta-score."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_COLS=['probability','meta_probability','ensemble_probability','cross_sectional_rank']

def build_stack(df):
    missing=[c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Stack missing required columns: {missing}')
    return df[BASE_COLS].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)

def expanding_stack(df:pd.DataFrame,min_train=200,step=25):
    d=df.copy().reset_index(drop=True); X=build_stack(d); y=pd.to_numeric(d['target_before_stop'],errors='coerce').fillna(0).astype(int); out=np.full(len(d),np.nan); disagreement=np.full(len(d),np.nan)
    for i in range(min_train,len(d),step):
        j=min(i+step,len(d)); train=X.iloc[:i].copy(); test=X.iloc[i:j].copy(); med=train.median().fillna(.5); train=train.fillna(med).fillna(.5); test=test.fillna(med).fillna(.5)
        if y.iloc[:i].nunique()<2:continue
        model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.25,max_iter=1000,class_weight='balanced'))]); model.fit(train,y.iloc[:i]); out[i:j]=model.predict_proba(test)[:,1]
        disagreement[i:j]=test[['probability','meta_probability','ensemble_probability']].std(axis=1).to_numpy()
    d['stack_probability']=out; d['model_disagreement']=disagreement; return d

def accept_stack(df,threshold=.62,max_disagreement=.12):
    return (df.stack_probability>=threshold)&(df.model_disagreement<=max_disagreement)
