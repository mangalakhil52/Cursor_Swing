#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.portfolio_risk import monte_carlo_risk

ap=argparse.ArgumentParser(); ap.add_argument('--input',default='reports/alpha_oos_predictions.csv'); ap.add_argument('--output',default='reports/portfolio_risk.csv'); ap.add_argument('--simulations',type=int,default=5000); ap.add_argument('--horizon',type=int,default=100); ap.add_argument('--block',type=int,default=5); a=ap.parse_args()
d=pd.read_csv(a.input).sort_values('date'); r=np.where(d.target_before_stop.astype(int).eq(1),2.,-1.)
# Research approximation: equal-sized sequential opportunities. Production portfolio simulator will use actual overlapping positions.
out=pd.DataFrame([monte_carlo_risk(r,a.simulations,a.horizon,a.block)]); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))
