#!/usr/bin/env python3
"""Convert final OOS model scores into executable, point-in-time portfolio candidates."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from src.ensemble_engine import ensemble_score

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/residual_alpha_oos.csv'); p.add_argument('--rank-input',default='reports/cross_sectional_rank.csv'); p.add_argument('--history',default='data/history'); p.add_argument('--output',default='reports/portfolio_candidates.csv'); p.add_argument('--probability-floor',type=float,default=.65); p.add_argument('--max-disagreement',type=float,default=.12); p.add_argument('--ensemble-threshold',type=float,default=68.); p.add_argument('--top-k',type=int,default=10); a=p.parse_args()

def naive_index(idx):
    x=pd.to_datetime(idx); return x.tz_localize(None) if getattr(x,'tz',None) is not None else x

d=pd.read_csv(a.input); d['date']=naive_index(d['date'])
if 'stack_probability' not in d: raise SystemExit('stack_probability missing from residual alpha input')
if 'model_disagreement' not in d: raise SystemExit('model_disagreement missing from residual alpha input')
prob=pd.to_numeric(d.get('residual_adjusted_probability',d['stack_probability']),errors='coerce'); d['execution_probability']=prob
rp=Path(a.rank_input)
if rp.exists():
    rank=pd.read_csv(rp,usecols=lambda c:c in {'date','symbol','cross_sectional_score','selected'}); rank['date']=naive_index(rank['date']); d=d.merge(rank.drop_duplicates(['date','symbol']),on=['date','symbol'],how='left')
else:d['cross_sectional_score']=np.nan
d['structural_score']=50.+50.*pd.to_numeric(d.get('cross_sectional_score',np.nan),errors='coerce').fillna(.5)
bench=Path(a.history)/'_benchmark.csv'
if not bench.exists(): raise SystemExit(f'Benchmark history missing: {bench}')
b=pd.read_csv(bench,index_col=0,parse_dates=True).sort_index(); b.index=naive_index(b.index); close=b['close'].astype(float); ret=close.pct_change()
def regime(dt):
    x=close.loc[close.index<=dt]; rr=ret.loc[ret.index<=dt]
    if len(x)<60:return 'SIDEWAYS'
    r20=x.iloc[-1]/x.iloc[-21]-1.; r60=x.iloc[-1]/x.iloc[-61]-1.; v20=rr.tail(20).std(); v120=rr.tail(120).std()
    if v20>1.5*max(v120,1e-9):return 'HIGH_VOL_SIDEWAYS'
    if r20>0.03 and r60>0.06:return 'STRONG_BULL'
    if r20>0.01 and r60>0.02:return 'BULL'
    if r20<-0.03 and r60<-0.06:return 'STRONG_BEAR'
    if r20<-0.01 and r60<-0.02:return 'BEAR'
    return 'SIDEWAYS'
d['regime']=d['date'].map(regime)
d['ensemble_score']=[ensemble_score(s,p,r) for s,p,r in zip(d.structural_score,d.execution_probability,d.regime)]
d['eligible']=(d.execution_probability>=a.probability_floor)&(d.model_disagreement<=a.max_disagreement)&(d.ensemble_score>=a.ensemble_threshold); d['date_rank']=d.groupby('date')['ensemble_score'].rank(method='first',ascending=False); d['eligible']&=d.date_rank<=a.top_k
rows=[]
for _,r in d[d.eligible & d.execution_probability.notna()].sort_values(['date','ensemble_score'],ascending=[True,False]).iterrows():
    f=Path(a.history)/(str(r.symbol)+'.csv')
    if not f.exists():continue
    bars=pd.read_csv(f,index_col=0,parse_dates=True).sort_index(); bars.index=naive_index(bars.index); dt=pd.Timestamp(r.date); px=bars.loc[bars.index<=dt]
    if px.empty:continue
    entry=float(px.close.iloc[-1]); volume=float(px.volume.iloc[-1]) if 'volume' in px else 0.; vol=float(px.close.pct_change().tail(20).std()*np.sqrt(252)) if len(px)>=21 else .2
    if not np.isfinite(entry) or entry<=0:continue
    rows.append({'entry_date':dt,'symbol':str(r.symbol),'direction':'LONG','entry_price':entry,'stop_price':entry*.96,'target1':entry*1.08,'target2':entry*1.16,'probability':float(r.execution_probability),'vol_annual':vol,'adv_value':entry*volume,'score':float(r.ensemble_score),'ensemble_score':float(r.ensemble_score),'regime':r.regime,'model_disagreement':float(r.model_disagreement),'signal_version':'STACK_EXEC_V1'})
out=pd.DataFrame(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f'eligible_candidates={len(out)} dates={out.entry_date.nunique() if len(out) else 0}')
