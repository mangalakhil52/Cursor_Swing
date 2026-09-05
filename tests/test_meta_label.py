import pandas as pd
from src.meta_label import FEATURES, build_features, meta_target, accept


def test_features_have_fixed_schema():
    d = pd.DataFrame({c: [0.5] for c in FEATURES})
    x = build_features(d)
    assert x.shape == (1, len(FEATURES))
    assert list(x.columns) == list(FEATURES)


def test_features_reject_missing_columns():
    d = pd.DataFrame({'probability': [.7], 'structural_score': [80]})
    try:
        build_features(d)
    except ValueError as exc:
        assert 'ensemble_probability' in str(exc)
    else:
        raise AssertionError('missing required features must fail loudly')


def test_accept_requires_both_models():
    d = pd.DataFrame({'probability': [.70, .70], 'structural_score': [80, 80]})
    x = accept(d, pd.Series([.60, .50]))
    assert list(x) == [True, False]


def test_meta_target_positive_r():
    d = pd.DataFrame({'r_multiple': [1., -1., 0.]})
    assert list(meta_target(d)) == [1, 0, 0]
