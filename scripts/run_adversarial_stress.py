#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.regime_adversary import stress_grid,adversarial_score

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/adversarial_stress.csv'); a=p.parse_args()
d=pd.read_csv(a.input); g=stress_grid(d); Path(a.output).parent.mkdir(parents=True,exist_ok=True); g.to_csv(a.output,index=False); pd.DataFrame([adversarial_score(g)]).to_csv(Path(a.output).with_name('adversarial_summary.csv'),index=False); print(adversarial_score(g))
