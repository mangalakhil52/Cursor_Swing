"""Leakage-aware ML ranking layer.

Uses sklearn when installed. This module is intentionally a ranking/probability
layer, not an autonomous trading system. Features must be point-in-time.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None
    IsotonicRegression = None

FEATURES = ["mom3","mom5","mom10","mom20","mom60","efficiency","ann_vol","volume_ratio","residual","score"]

@dataclass
class MLResult:
    probability: float
    model_score: float
    trained: bool

class PurgedWalkForward:
    """Chronological folds with an embargo measured in observations."""
    def __init__(self, n_splits: int = 5, test_size: int = 60, embargo: int = 10):
        self.n_splits=n_splits; self.test_size=test_size; self.embargo=embargo

    def split(self, n: int):
        for k in range(self.n_splits):
            test_end=n-k*self.test_size
            test_start=test_end-self.test_size
            train_end=max(0,test_start-self.embargo)
            if test_start<=0 or train_end<30: continue
            yield np.arange(train_end), np.arange(test_start,test_end)

def train_oos(results: pd.DataFrame, target_col: str = "target_hit", n_splits: int = 5, embargo: int = 10) -> pd.DataFrame:
    if HistGradientBoostingClassifier is None:
        raise RuntimeError("scikit-learn is required for ML research: pip install scikit-learn")
    x=results.copy().sort_values(["execution_date","symbol"]).reset_index(drop=True)
    x=x.dropna(subset=FEATURES+[target_col]).reset_index(drop=True)
    x["ml_probability"]=np.nan
    splitter=PurgedWalkForward(n_splits=n_splits,test_size=max(20,len(x)//(n_splits+2)),embargo=embargo)
    for train_idx,test_idx in splitter.split(len(x)):
        model=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.04,l2_regularization=2.0,random_state=42)
        model.fit(x.loc[train_idx,FEATURES],x.loc[train_idx,target_col].astype(int))
        x.loc[test_idx,"ml_probability"]=model.predict_proba(x.loc[test_idx,FEATURES])[:,1]
    return x

def rank_latest(results: pd.DataFrame) -> pd.DataFrame:
    x=results.copy()
    if "ml_probability" not in x:return x
    return x.sort_values(["execution_date","ml_probability","score"],ascending=[False,False,False]).reset_index(drop=True)
