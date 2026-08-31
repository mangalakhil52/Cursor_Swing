#!/usr/bin/env python3
"""Run regime calibration without requiring a precomputed regime column."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from src.regime_calibration import by_regime, gate


def ensure_regime(d: pd.DataFrame) -> pd.DataFrame:
    if 'regime' in d.columns:
        return d
    # The alpha dataset currently contains signal-time trend/volatility features,
    # but no categorical regime. Derive a conservative point-in-time proxy from
    # EMA spread and realized volatility; never use forward labels.
    ema = pd.to_numeric(d.get('ema_spread', pd.Series(0.0, index=d.index)), errors='coerce').fillna(0.0)
    vol = pd.to_numeric(d.get('vol_20', pd.Series(0.0, index=d.index)), errors='coerce').fillna(0.0)
    q = float(vol.quantile(.75)) if vol.notna().any() else 0.0
    d = d.copy()
    d['regime'] = np.select(
        [
            (ema >= .01) & (vol <= q),
            (ema <= -.01) & (vol <= q),
            (vol > q),
        ],
        ['BULL', 'BEAR', 'HIGH_VOL'],
        default='SIDEWAYS'
    )
    return d

p=argparse.ArgumentParser()
p.add_argument('--input',default='reports/alpha_oos_predictions.csv')
p.add_argument('--output',default='reports/regime_calibration.csv')
a=p.parse_args()
d=ensure_regime(pd.read_csv(a.input))
rows=by_regime(d)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
rows.to_csv(a.output,index=False)
pd.DataFrame([gate(rows)]).to_csv(Path(a.output).with_name('regime_calibration_gate.csv'),index=False)
print(gate(rows))
