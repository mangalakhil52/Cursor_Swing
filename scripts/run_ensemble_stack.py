#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.ensemble_stacker import expanding_stack,accept_stack

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_ensemble_oos.csv'); p.add_argument('--output',default='reports/ensemble_stack_oos.csv'); a=p.parse_args()
d=pd.read_csv(a.input); out=expanding_stack(d); out['stack_accept']=accept_stack(out); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f'rows={len(out)} scored={out.stack_probability.notna().sum()} accepted={int(out.stack_accept.sum())}')
