"""Search for incremental alpha left unexplained by the existing stack."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES=[
    'probability','stack_probability','ensemble_probability','model_disagreement',
    'structural_score','relative_strength','residual_momentum','regime_fit',
    'volatility_efficiency','atr_pct','volume_ratio','efficiency_20',
    'ema9_gap','ema21_gap','ema_spread','ret_5','ret_20','ret_60'
]

def build_features(d):
    missing=[c for c in FEATURES if c not in d.columns]
    if missing:
        raise ValueError(f'Residual alpha missing required features: {missing}')
    return d[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)

def expanding_residual_alpha(df:pd.DataFrame,min_train=250,step=25):
    d=df.copy().reset_index(drop=True); X=build_features(d)
    y=pd.to_numeric(d['target_before_stop'],errors='coerce').fillna(0.).astype(float)
    base=pd.to_numeric(d.get('stack_probability',d.get('ensemble_probability',d.get('probability',.5))),errors='coerce').fillna(.5)
    residual=np.full(len(d),np.nan)
    for i in range(min_train,len(d),step):
        j=min(i+step,len(d)); med=X.iloc[:i].median().fillna(0.)
        train=X.iloc[:i].fillna(med).fillna(0.); test=X.iloc[i:j].fillna(med).fillna(0.)
        yy=y.iloc[:i]-base.iloc[:i]
        model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1.,min_samples_leaf=20,random_state=23)
        model.fit(train,yy); residual[i:j]=model.predict(test)
    d['residual_alpha']=residual
    d['residual_adjusted_probability']=np.clip(base+np.nan_to_num(residual,nan=0.)*.35,0,1)
    return d
