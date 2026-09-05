"""Compare incumbent Alpha with the complete executable research stack."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.ensemble_engine import ensemble_score


def simulate(d, pcol, threshold=.60, cost_bps=10.0):
    x=d[pd.to_numeric(d[pcol],errors='coerce')>=threshold].copy()
    return _metrics(x,cost_bps)


def _metrics(x,cost_bps=10.0):
    if x.empty:
        return {'trades':0,'win_rate':0.,'expectancy_r':0.,'profit_factor':0.,'max_drawdown_r':0.,'total_r':0.}
    r=np.where(pd.to_numeric(x.target_before_stop,errors='coerce').fillna(0).astype(int).eq(1),2.0,-1.0)
    r=r-(cost_bps/10000.0)/.04
    equity=np.cumsum(r); peak=np.maximum.accumulate(np.r_[0.,equity]); dd=equity-peak[1:]
    wins=r[r>0].sum(); losses=-r[r<0].sum()
    return {'trades':int(len(r)),'win_rate':float((r>0).mean()),'expectancy_r':float(r.mean()),'profit_factor':float(wins/losses) if losses else float('inf'),'max_drawdown_r':float(dd.min()),'total_r':float(r.sum())}


def final_candidates(d,probability_floor=.65,max_disagreement=.12,ensemble_threshold=68.,top_k=10):
    x=d.copy()
    x['execution_probability']=pd.to_numeric(x.get('residual_adjusted_probability'),errors='coerce')
    x['model_disagreement']=pd.to_numeric(x.get('model_disagreement'),errors='coerce')
    x['structural_score_100']=100.*pd.to_numeric(x.get('structural_score'),errors='coerce')
    x['ensemble_score']=pd.to_numeric(x.get('ensemble_score'),errors='coerce')
    missing=[]
    for c in ['execution_probability','model_disagreement','structural_score_100','market_regime']:
        if c not in x or x[c].isna().all(): missing.append(c)
    if missing: raise ValueError(f'Final comparison missing required columns: {missing}')
    x['ensemble_score']=[ensemble_score(s,p,r) for s,p,r in zip(x.structural_score_100,x.execution_probability,x.market_regime)]
    x['eligible']=(x.execution_probability>=probability_floor)&(x.model_disagreement<=max_disagreement)&(x.ensemble_score>=ensemble_threshold)
    x['date_rank']=x.groupby('date')['ensemble_score'].rank(method='first',ascending=False)
    return x[x.eligible & (x.date_rank<=top_k)].copy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='reports/alpha_oos_predictions.csv')
    ap.add_argument('--final-input',default=None)
    ap.add_argument('--output',default='reports/strategy_comparison.csv')
    ap.add_argument('--thresholds',default='0.55,0.60,0.65,0.70,0.75')
    ap.add_argument('--cost-bps',type=float,default=10.0)
    ap.add_argument('--final-probability-floor',type=float,default=.65)
    ap.add_argument('--final-max-disagreement',type=float,default=.12)
    ap.add_argument('--final-ensemble-threshold',type=float,default=68.)
    ap.add_argument('--final-top-k',type=int,default=10)
    a=ap.parse_args()
    d=pd.read_csv(a.input); rows=[]
    base=d.copy(); base['base_prob']=base.target_before_stop.mean()
    for name,pcol,t in [('BASE','base_prob',0.0)]+[(f'ALPHA_{t:.2f}','probability',t) for t in map(float,a.thresholds.split(','))]:
        m=simulate(base,pcol,t,a.cost_bps); m['strategy']=name; m['threshold']=t; rows.append(m)
    if a.final_input:
        f=pd.read_csv(a.final_input)
        f['date']=pd.to_datetime(f['date'])
        selected=final_candidates(f,a.final_probability_floor,a.final_max_disagreement,a.final_ensemble_threshold,a.final_top_k)
        m=_metrics(selected,a.cost_bps); m['strategy']='FINAL_STACK_EXEC'; m['threshold']=a.final_probability_floor; m['selected_dates']=int(selected.date.nunique()); rows.append(m)
        print(f'final_selected={len(selected)} final_dates={selected.date.nunique()}')
    out=pd.DataFrame(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))

if __name__=='__main__': main()
