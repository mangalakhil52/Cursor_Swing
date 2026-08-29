#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.residual_alpha import expanding_residual_alpha

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/ensemble_stack_oos.csv'); p.add_argument('--output',default='reports/residual_alpha_oos.csv'); a=p.parse_args()
d=pd.read_csv(a.input); out=expanding_residual_alpha(d); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f'rows={len(out)} scored={out.residual_alpha.notna().sum()}')
