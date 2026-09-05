"""Meta-labeling: estimate conditional quality of an existing alpha signal."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURES=(
    'probability','ensemble_probability','ensemble_disagreement','structural_score',
    'relative_strength','residual_momentum','volatility_efficiency','regime_fit',
    'atr_pct','volume_ratio','efficiency_20','ema_spread','ret_5','ret_20','ret_60'
)

@dataclass(frozen=True)
class MetaConfig:
    min_probability: float=.62
    min_meta_probability: float=.58
    min_expected_r: float=.10

def build_features(df:pd.DataFrame)->pd.DataFrame:
    missing=[c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f'Meta-label missing required features: {missing}')
    return df.loc[:,FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)

def meta_target(df:pd.DataFrame)->pd.Series:
    """Positive when the original signal's realized outcome clears a 0R hurdle."""
    if 'r_multiple' in df:return (pd.to_numeric(df.r_multiple,errors='coerce')>0).astype(int)
    return pd.to_numeric(df.get('target_before_stop',0),errors='coerce').fillna(0).astype(int)

def accept(df:pd.DataFrame, meta_probability:pd.Series, cfg:MetaConfig=MetaConfig())->pd.Series:
    p=pd.to_numeric(df.probability,errors='coerce').fillna(0.)
    mp=pd.Series(meta_probability,index=df.index).fillna(0.)
    return (p>=cfg.min_probability)&(mp>=cfg.min_meta_probability)
