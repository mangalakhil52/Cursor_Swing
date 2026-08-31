"""Learn cross-sectional opportunity scores using strictly prior dates."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES=('probability','structural_score','residual_momentum','relative_strength','volatility_efficiency','regime_fit','rsi','atr_pct','volume_ratio')

def prepare(df):
    x=df.copy()
    for c in FEATURES:
        if c not in x:x[c]=0.
        x[c]=pd.to_numeric(x[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        x[c]=x[c].fillna(x[c].median() if x[c].notna().any() else 0.)
    return x

def walk_forward_score(df:pd.DataFrame,date_col='date',outcome_col='target_before_stop',min_train_dates=30)->pd.DataFrame:
    x=prepare(df).sort_values([date_col,'symbol']).reset_index(drop=True); x['learned_alpha']=np.nan
    dates=sorted(pd.to_datetime(x[date_col]).dropna().unique())
    for i,dt in enumerate(dates):
        if i<min_train_dates: continue
        train=x[pd.to_datetime(x[date_col])<dt]; test_idx=x.index[pd.to_datetime(x[date_col])==dt]
        y=pd.to_numeric(train[outcome_col],errors='coerce').fillna(0).astype(int)
        if len(train)<2 or y.nunique()<2: continue
        model=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=7,l2_regularization=2.,min_samples_leaf=max(2,min(20,len(train)//10)),random_state=42)
        model.fit(train.loc[:,FEATURES],y); x.loc[test_idx,'learned_alpha']=model.predict_proba(x.loc[test_idx,FEATURES])[:,1]
    x['learned_alpha_rank']=x.groupby(date_col).learned_alpha.rank(pct=True,method='average')
    return x
