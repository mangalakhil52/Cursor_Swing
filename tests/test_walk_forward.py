import pandas as pd
from src.walk_forward import make_folds, split_frame
from src.alpha_calibration import calibration_table, brier_score

def test_folds_are_time_ordered_and_embargoed():
    f=make_folds(1000,train_size=500,test_size=100,step=100,embargo=10)
    assert f and all(x.train_end <= x.test_start-10 for x in f)

def test_split_is_disjoint():
    d=pd.DataFrame({'x':range(700)})
    tr,te=split_frame(d,make_folds(700,500,100,100,10)[0])
    assert tr.index.max()<te.index.min()

def test_calibration_metrics():
    p=[.1,.2,.8,.9]; y=[0,0,1,1]
    assert brier_score(p,y)<.05
    assert len(calibration_table(p,y,bins=4))>0
