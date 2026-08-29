#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.ensemble_stacker import expanding_stack,accept_stack

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/meta_label_oos.csv'); p.add_argument('--output',default='reports/ensemble_stack_oos.csv'); p.add_argument('--rank-input',default='reports/cross_sectional_rank.csv'); a=p.parse_args()
d=pd.read_csv(a.input).sort_values(['date','symbol']).reset_index(drop=True)
# Bring the cross-sectional rank into the same OOS frame before stacking. The join is
# point-in-time: date+symbol only, with no future observations introduced.
r=Path(a.rank_input)
if r.exists():
    rank=pd.read_csv(r,usecols=lambda c: c in {'date','symbol','cross_sectional_rank','cross_sectional_score','selected'})
    rank['date']=pd.to_datetime(rank['date']); d['date']=pd.to_datetime(d['date'])
    d=d.merge(rank.drop_duplicates(['date','symbol']),on=['date','symbol'],how='left')
out=expanding_stack(d); out['stack_accept']=accept_stack(out)
Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
print(f'rows={len(out)} scored={out.stack_probability.notna().sum()} accepted={int(out.stack_accept.sum())}')
