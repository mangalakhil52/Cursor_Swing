import pandas as pd
from src.meta_label import build_features,meta_target,accept

def test_features_have_fixed_schema():
    d=pd.DataFrame({'probability':[.7],'structural_score':[80]})
    assert build_features(d).shape==(1,8)

def test_accept_requires_both_models():
    d=pd.DataFrame({'probability':[.70,.70],'structural_score':[80,80]})
    x=accept(d,pd.Series([.60,.50]))
    assert list(x)==[True,False]

def test_meta_target_positive_r():
    d=pd.DataFrame({'r_multiple':[1.,-1.,0.]})
    assert list(meta_target(d))==[1,0,0]
