#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.ensemble_engine import ensemble_score
from src.ensemble_metrics import metrics, promotion_gate

ap=argparse.ArgumentParser(); ap.add_argument('--input',default='reports/alpha_oos_predictions.csv'); ap.add_argument('--output',default='reports/ensemble_oos_comparison.csv'); a=ap.parse_args()
d=pd.read_csv(a.input)
if 'structural_score' not in d: d['structural_score']=d.get('score',68.0)
if 'regime' not in d: d['regime']='SIDEWAYS'
rows=[]
for mode in ('BASE','ENSEMBLE'):
    if mode=='BASE': take=d.structural_score>=68
    else:
        d['ensemble_score']=[ensemble_score(s,p,r) for s,p,r in zip(d.structural_score,d.probability,d.regime)]
        take=(d.ensemble_score>=68)&(d.probability>=.60)
    r=pd.Series(np.where(d.target_before_stop.astype(int).eq(1),2.,-1.),index=d.index)[take]
    m=metrics(r); m['strategy']=mode; rows.append(m)
out=pd.DataFrame(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
decision=promotion_gate(rows[0],rows[1]); pd.DataFrame([decision]).to_csv(Path(a.output).with_name('ensemble_promotion_gate.csv'),index=False)
print(out.to_string(index=False)); print('PROMOTION',decision)
