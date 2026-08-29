"""Purged walk-forward evaluation helpers.

Train windows precede validation windows. An embargo removes observations immediately
following the validation boundary, reducing leakage from overlapping forward labels.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Fold:
    train_start:int; train_end:int; test_start:int; test_end:int

def make_folds(n:int, train_size:int=500, test_size:int=100, step:int=100, embargo:int=10):
    folds=[]; start=train_size
    while start+test_size<=n:
        test_end=start+test_size
        train_end=max(0,start-embargo)
        if train_end>0: folds.append(Fold(0,train_end,start,test_end))
        start+=step
    return folds

def split_frame(df:pd.DataFrame, fold:Fold):
    return df.iloc[fold.train_start:fold.train_end].copy(), df.iloc[fold.test_start:fold.test_end].copy()
