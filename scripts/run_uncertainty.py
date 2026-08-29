#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.uncertainty import uncertainty,calibration_bins

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/alpha_uncertainty.csv'); a=p.parse_args()
d=pd.read_csv(a.input); m=uncertainty(d.target_before_stop,d.probability); Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame([m]).to_csv(a.output,index=False); calibration_bins(d.target_before_stop,d.probability).to_csv(Path(a.output).with_name('alpha_calibration_bins.csv'),index=False); print(m)
