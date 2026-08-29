"""Leakage-safe ensemble of structurally different alpha estimators."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES=['probability','structural_score','residual_momentum','relative_strength','volatility_efficiency','regime_fit','rsi','atr_pct','volume_ratio']

def _features(d):
    x=d.reindex(columns=FEATURES).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan); return x.fillna(x.median()).fillna(0.)

def expanding_ensemble(df:pd.DataFrame, min_train=150, step=25):
    d=df.copy().reset_index(drop=True); X=_features(d); y=pd.to_numeric(d['target_before_stop'],errors='coerce').fillna(0).astype(int); out=np.full(len(d),np.nan)
    for i in range(min_train,len(d),step):
        j=min(i+step,len(d));
        if y.iloc[:i].nunique()<2: continue
        lin=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.5,max_iter=1000,class_weight='balanced'))]); tree=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1.0,min_samples_leaf=20,random_state=17)
        lin.fit(X.iloc[:i],y.iloc[:i]); tree.fit(X.iloc[:i],y.iloc[:i]); out[i:j]=.5*lin.predict_proba(X.iloc[i:j])[:,1]+.5*tree.predict_proba(X.iloc[i:j])[:,1]
    d['ensemble_probability']=out; d['ensemble_disagreement']=np.nan
    return d
