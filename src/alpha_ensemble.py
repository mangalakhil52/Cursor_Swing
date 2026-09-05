"""Leakage-safe ensemble of structurally different alpha estimators."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES=[
    'probability','structural_score','residual_momentum','relative_strength',
    'volatility_efficiency','regime_fit','atr_pct','volume_ratio',
    'efficiency_20','ema9_gap','ema21_gap','ema_spread','ret_5','ret_20','ret_60'
]

def _features(d):
    # Imputation is deliberately deferred to each expanding training window.
    return d.reindex(columns=FEATURES).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)

def _models():
    lin=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.5,max_iter=1000,class_weight='balanced'))])
    tree=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1.0,min_samples_leaf=20,random_state=17)
    return lin,tree

def expanding_ensemble(df:pd.DataFrame,min_train=150,step=25,retrain_every=100):
    """Expanding OOS ensemble with periodic refits and training-only imputation."""
    if min_train < 2 or step <= 0 or retrain_every <= 0:
        raise ValueError('min_train must be >=2; step and retrain_every must be positive')
    d=df.copy().reset_index(drop=True)
    X=_features(d)
    y=pd.to_numeric(d['target_before_stop'],errors='coerce').fillna(0).astype(int)
    out=np.full(len(d),np.nan); disagreement=np.full(len(d),np.nan)
    lin=tree=None; med=pd.Series(0.,index=X.columns); next_refit=min_train; i=min_train
    while i < len(d):
        j=min(i+step,len(d))
        if lin is None or i >= next_refit:
            if y.iloc[:i].nunique() >= 2:
                med=X.iloc[:i].median().fillna(0.)
                train=X.iloc[:i].fillna(med).fillna(0.)
                lin,tree=_models(); lin.fit(train,y.iloc[:i]); tree.fit(train,y.iloc[:i])
                next_refit=i+retrain_every
        if lin is not None:
            test=X.iloc[i:j].fillna(med).fillna(0.)
            p_lin=lin.predict_proba(test)[:,1]; p_tree=tree.predict_proba(test)[:,1]
            out[i:j]=.5*p_lin+.5*p_tree; disagreement[i:j]=np.abs(p_lin-p_tree)
        i=j
    d['ensemble_probability']=out; d['ensemble_disagreement']=disagreement
    return d
