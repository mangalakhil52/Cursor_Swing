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
    x=d.reindex(columns=FEATURES).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    return x.fillna(x.median()).fillna(0.)

def _models():
    lin=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.5,max_iter=1000,class_weight='balanced'))])
    tree=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1.0,min_samples_leaf=20,random_state=17)
    return lin,tree

def expanding_ensemble(df:pd.DataFrame,min_train=150,step=25,retrain_every=100):
    """Expanding OOS ensemble with periodic refits.

    Models are trained only on observations strictly before the current
    prediction window. Between refits the already-trained models are reused,
    preserving OOS discipline while reducing expensive tree fits substantially.
    ``retrain_every`` is measured in prediction rows, not calendar days.
    """
    if min_train < 2 or step <= 0 or retrain_every <= 0:
        raise ValueError('min_train must be >=2; step and retrain_every must be positive')
    d=df.copy().reset_index(drop=True)
    X=_features(d)
    y=pd.to_numeric(d['target_before_stop'],errors='coerce').fillna(0).astype(int)
    out=np.full(len(d),np.nan)
    lin=tree=None
    next_refit=min_train
    i=min_train
    while i < len(d):
        j=min(i+step,len(d))
        if lin is None or i >= next_refit:
            if y.iloc[:i].nunique() >= 2:
                lin,tree=_models()
                lin.fit(X.iloc[:i],y.iloc[:i])
                tree.fit(X.iloc[:i],y.iloc[:i])
                next_refit=i+retrain_every
        if lin is not None:
            out[i:j]=.5*lin.predict_proba(X.iloc[i:j])[:,1]+.5*tree.predict_proba(X.iloc[i:j])[:,1]
        i=j
    d['ensemble_probability']=out
    d['ensemble_disagreement']=np.nan
    return d
