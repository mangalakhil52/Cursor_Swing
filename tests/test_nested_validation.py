import pandas as pd
from src.nested_validation import make_nested_folds,choose_threshold

def test_nested_folds_are_ordered():
    f=make_nested_folds(20000,10000,2000,2000,2000,10)[0]
    assert f.train_end+10<=f.validation_start
    assert f.validation_end<=f.test_start

def test_threshold_selection_uses_validation():
    d=pd.DataFrame({'probability':[.55,.60,.65,.70]*20,'target_before_stop':[0,0,1,1]*20})
    x=choose_threshold(d,d,thresholds=(.55,.65,.70))
    assert x['threshold'] in (.65,.70)
