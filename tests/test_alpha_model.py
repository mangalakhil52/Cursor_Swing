import numpy as np
import pandas as pd

from src.alpha_model import build_features, build_label, make_research_frame


def sample_data(n=120):
    idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="Asia/Kolkata")
    close = 100 * np.exp(np.cumsum(np.full(n, 0.002)))
    return pd.DataFrame({
        "open": close * .995,
        "high": close * 1.01,
        "low": close * .99,
        "close": close,
        "volume": np.full(n, 200000.),
    }, index=idx)


def test_features_use_current_history_only():
    d = sample_data()
    a = build_features(d.iloc[:100])
    b = build_features(d.iloc[:101])
    assert a["ret_5"] == b["ret_5"]
    assert a["ret_20"] == b["ret_20"]


def test_label_is_forward_looking():
    d = sample_data()
    label = build_label(d, 80, target_pct=.01, stop_pct=.20)
    assert label is not None
    assert label.forward_return_5 > 0
    assert label.forward_return_10 > 0
    assert label.target_before_stop == 1


def test_research_frame_has_no_future_feature_columns():
    d = sample_data()
    frame = make_research_frame(d)
    assert len(frame) > 0
    assert "forward_return_5" in frame.columns
    assert "forward_return_10" in frame.columns
    assert "target_before_stop" in frame.columns
