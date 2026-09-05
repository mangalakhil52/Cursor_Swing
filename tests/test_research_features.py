import numpy as np
import pandas as pd
import pytest

from scripts.build_research_features import build_features
from src.alpha_ensemble import expanding_ensemble
from src.meta_label import build_features as build_meta_features
from src.residual_alpha import build_features as build_residual_features


def test_research_features_are_constructed_from_available_columns():
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    bench = pd.DataFrame({'close': np.linspace(100, 110, len(dates))}, index=dates)
    rows = []
    for symbol, bump in [('AAA', 0.0), ('BBB', 0.01)]:
        for i, dt in enumerate(dates):
            rows.append({
                'date': dt, 'symbol': symbol, 'probability': .5,
                'target_before_stop': i % 2, 'ret_5': .01 + bump,
                'ret_20': .04 + bump, 'ret_60': .10 + bump,
                'efficiency_20': .4, 'ema_spread': .01,
                'range_position_20': .7, 'atr_pct': .02,
                'volume_ratio': 1.2,
            })
    out = build_features(pd.DataFrame(rows), bench)
    for c in ['structural_score','residual_momentum','relative_strength','volatility_efficiency','regime_fit','market_regime']:
        assert c in out.columns
    assert out['structural_score'].between(0, 1).all()
    assert out['regime_fit'].between(0, 1).all()


def test_downstream_feature_contracts_fail_loudly():
    with pytest.raises(ValueError):
        build_meta_features(pd.DataFrame({'probability':[.5]}))
    with pytest.raises(ValueError):
        build_residual_features(pd.DataFrame({'probability':[.5]}))


def test_ensemble_reports_model_disagreement():
    n = 220
    d = pd.DataFrame({
        'target_before_stop': np.arange(n) % 2,
        'probability': np.linspace(.2, .8, n),
        'structural_score': .5,
        'residual_momentum': np.sin(np.arange(n)/10),
        'relative_strength': np.cos(np.arange(n)/15),
        'volatility_efficiency': 10., 'regime_fit': .5,
        'atr_pct': .02, 'volume_ratio': 1.0, 'efficiency_20': .3,
        'ema9_gap': .01, 'ema21_gap': .02, 'ema_spread': .01,
        'ret_5': .01, 'ret_20': .03, 'ret_60': .08,
    })
    out = expanding_ensemble(d, min_train=150, step=25, retrain_every=50)
    assert out['ensemble_probability'].notna().sum() > 0
    assert out['ensemble_disagreement'].notna().sum() > 0
