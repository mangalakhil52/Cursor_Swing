#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.sensitivity_analysis import evaluate_grid,plateau_score

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/alpha_oos_predictions.csv'); p.add_argument('--output',default='reports/parameter_sensitivity.csv'); a=p.parse_args()
d=pd.read_csv(a.input)
if 'structural_score' not in d:d['structural_score']=d.get('score',68.)
grid=evaluate_grid(d); Path(a.output).parent.mkdir(parents=True,exist_ok=True); grid.to_csv(a.output,index=False); pd.DataFrame([plateau_score(grid)]).to_csv(Path(a.output).with_name('parameter_plateau.csv'),index=False); print(plateau_score(grid))
