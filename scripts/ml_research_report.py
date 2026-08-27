"""Run OOS ML research on the deterministic research dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from src.ml_ranker import train_oos
from src.calibration import brier_score, calibration_gap, isotonic_like_bins

ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
df=pd.read_csv(args.input)
# target_hit is already defined only from future bars by research_engine.
out=train_oos(df)
out.to_csv(args.out,index=False)
valid=out.dropna(subset=['ml_probability']).copy()
y=valid['target_hit'].astype(float); p=valid['ml_probability'].astype(float)
summary={'rows':len(out),'oos_rows':len(valid),'brier_score':brier_score(y,p),'calibration_gap':calibration_gap(y,p)}
Path('reports').mkdir(exist_ok=True)
Path('reports/research_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('\nCALIBRATION BINS\n')
print(isotonic_like_bins(y,p).to_string(index=False))
