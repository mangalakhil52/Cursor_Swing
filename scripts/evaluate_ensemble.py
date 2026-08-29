#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.ensemble_evaluator import evaluate

ap=argparse.ArgumentParser(); ap.add_argument('--input',default='reports/alpha_oos_predictions.csv'); ap.add_argument('--output',default='reports/ensemble_oos_comparison.csv'); a=ap.parse_args()
d=pd.read_csv(a.input)
if 'structural_score' not in d.columns: d['structural_score']=d.get('score',68.0)
if 'regime' not in d.columns: d['regime']='SIDEWAYS'
out=evaluate(d)
Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))
