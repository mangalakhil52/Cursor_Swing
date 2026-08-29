#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.regime_calibration import by_regime,gate

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/regime_calibration.csv'); a=p.parse_args()
d=pd.read_csv(a.input); rows=by_regime(d); Path(a.output).parent.mkdir(parents=True,exist_ok=True); rows.to_csv(a.output,index=False); pd.DataFrame([gate(rows)]).to_csv(Path(a.output).with_name('regime_calibration_gate.csv'),index=False); print(gate(rows))
