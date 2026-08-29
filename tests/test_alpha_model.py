import numpy as np
import pandas as pd
from src.alpha_model import build_features, build_label, make_research_frame

def sample_data(n=120):
    idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="Asia/Kolkata")
    close = 100 * np.exp(np.cumsum(np.full(n, 0.002)))
    return pd.DataFrame({"open":close*.995,"high":close*1.01,"low":close*.99,"close":close,"volume":np.full(n,200000.)}, index=idx)

def test_features_are_finite():
    f=build_features(sample_data())
    assert np.isfinite(f["ret_5"]); assert np.isfinite(f["efficiency_20"])

def test_label_is_forward_looking():
    x=build_label(sample_data(),80,target_pct=.01,stop_pct=.20)
    assert x is not None and x.forward_return_5>0 and x.forward_return_10>0 and x.target_before_stop==1

def test_research_frame_contains_labels():
    f=make_research_frame(sample_data())
    assert len(f)>0
    assert {"forward_return_5","forward_return_10","target_before_stop"}.issubset(f.columns)
