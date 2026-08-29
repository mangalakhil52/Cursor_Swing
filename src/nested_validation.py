"""Nested chronological validation for selecting strategy parameters without test leakage."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class NestedFold:
    train_start:int; train_end:int; validation_start:int; validation_end:int; test_start:int; test_end:int

def make_nested_folds(n:int, train_size:int=12000, validation_size:int=2000, test_size:int=2000, step:int=2000, embargo:int=10):
    out=[]; test_start=train_size+validation_size
    while test_start+test_size<=n:
        validation_start=train_size; validation_end=validation_start+validation_size
        train_end=max(0,validation_start-embargo)
        if train_end>0:
            out.append(NestedFold(0,train_end,validation_start,validation_end,test_start,test_start+test_size))
        train_size += step; test_start += step
    return out

def choose_threshold(train:pd.DataFrame, validation:pd.DataFrame, prob_col='probability', outcome_col='target_before_stop', thresholds=(.55,.60,.65,.70,.75)):
    best=None
    for t in thresholds:
        x=validation[validation[prob_col]>=t]
        if len(x)<20: continue
        r=np.where(x[outcome_col].astype(int).eq(1),2.,-1.)
        score=float(r.mean())
        if best is None or score>best['validation_expectancy_r']: best={'threshold':float(t),'validation_expectancy_r':score,'validation_trades':int(len(x))}
    return best
