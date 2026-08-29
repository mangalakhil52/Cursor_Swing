import pandas as pd
from src.walk_forward import make_folds

def test_default_fold_structure():
    folds=make_folds(20000,10000,2000,2000,10)
    assert len(folds)>0
    assert all(f.train_end<=f.test_start-10 for f in folds)
    assert all(f.test_end>f.test_start for f in folds)
