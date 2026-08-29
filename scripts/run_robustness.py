#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.robustness import parameter_stability
p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/parameter_sensitivity.csv'); p.add_argument('--output',default='reports/robustness_summary.csv'); a=p.parse_args()
d=pd.read_csv(a.input); out=pd.DataFrame([parameter_stability(d)]); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))
