"""Search for incremental alpha left unexplained by the existing stack."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES=['rsi','atr_pct','volume_ratio','relative_strength','residual_momentum','regime_score','distance_to_resistance','distance_to_support','structural_score']

def build_features(d):
    x=d.reindex(columns=FEATURES).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan); return x.fillna(x.median()).fillna(0.)

def expanding_residual_alpha(df:pd.DataFrame,min_train=250,step=25):
    d=df.copy().reset_index(drop=True); X=build_features(d); y=pd.to_numeric(d['target_before_stop'],errors='coerce').fillna(0.).astype(float)
    base=pd.to_numeric(d.get('stack_probability',d.get('ensemble_probability',d.get('probability',.5))),errors='coerce').fillna(.5)
    residual=np.full(len(d),np.nan)
    for i in range(min_train,len(d),step):
        j=min(i+step,len(d)); yy=y.iloc[:i]-base.iloc[:i]
        model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1.,min_samples_leaf=20,random_state=23)
        model.fit(X.iloc[:i],yy); residual[i:j]=model.predict(X.iloc[i:j])
    d['residual_alpha']=residual; d['residual_adjusted_probability']=np.clip(base+np.nan_to_num(residual,nan=0.)*.35,0,1); return d
