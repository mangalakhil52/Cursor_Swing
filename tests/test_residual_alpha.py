import pandas as pd
from src.residual_alpha import FEATURES, expanding_residual_alpha


def test_residual_is_future_blind():
    n = 300
    d = pd.DataFrame({c: 0.5 for c in FEATURES}, index=range(n))
    d['probability'] = [.6, .8] * (n // 2)
    d['stack_probability'] = [.55, .75] * (n // 2)
    d['ensemble_probability'] = [.58, .78] * (n // 2)
    d['model_disagreement'] = [.05, .10] * (n // 2)
    d['target_before_stop'] = [0, 1] * (n // 2)
    d['rsi'] = [40, 70] * (n // 2) if 'rsi' in d else 0
    d['atr_pct'] = [.02, .03] * (n // 2)
    d['volume_ratio'] = [1., 2.] * (n // 2)
    d['relative_strength'] = [.2, .8] * (n // 2)
    d['residual_momentum'] = [.1, .7] * (n // 2)
    d['regime_fit'] = [.5, .9] * (n // 2)
    d['volatility_efficiency'] = [10., 30.] * (n // 2)
    d['efficiency_20'] = [.3, .8] * (n // 2)
    d['ema9_gap'] = [.01, .03] * (n // 2)
    d['ema21_gap'] = [.01, .04] * (n // 2)
    d['ema_spread'] = [.01, .03] * (n // 2)
    d['ret_5'] = [.01, .04] * (n // 2)
    d['ret_20'] = [.02, .08] * (n // 2)
    d['ret_60'] = [.03, .12] * (n // 2)
    x = expanding_residual_alpha(d, min_train=250, step=25)
    assert x.residual_alpha.iloc[:250].isna().all()
    assert x.residual_alpha.iloc[250:].notna().any()
