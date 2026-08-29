"""Meta-labeling: learn whether an existing signal is worth taking.

The primary model proposes direction/edge; this layer estimates conditional trade quality.
It intentionally supports simple, auditable features and out-of-sample training only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURES=('probability','structural_score','rsi','atr_pct','volume_ratio','distance_to_resistance','distance_to_support','regime_score')
@dataclass(frozen=True)
class MetaConfig:
    min_probability: float=.62
    min_meta_probability: float=.58
    min_expected_r: float=.10

def build_features(df:pd.DataFrame)->pd.DataFrame:
    x=df.copy()
    for c in FEATURES:
        if c not in x:x[c]=0.
    return x.loc[:,FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0.)

def meta_target(df:pd.DataFrame)->pd.Series:
    """Positive only when the original signal's realized outcome clears a 0R hurdle."""
    if 'r_multiple' in df:return (pd.to_numeric(df.r_multiple,errors='coerce')>0).astype(int)
    return pd.to_numeric(df.get('target_before_stop',0),errors='coerce').fillna(0).astype(int)

def accept(df:pd.DataFrame, meta_probability:pd.Series, cfg:MetaConfig=MetaConfig())->pd.Series:
    p=pd.to_numeric(df.probability,errors='coerce').fillna(0.)
    mp=pd.Series(meta_probability,index=df.index).fillna(0.)
    return (p>=cfg.min_probability)&(mp>=cfg.min_meta_probability)
