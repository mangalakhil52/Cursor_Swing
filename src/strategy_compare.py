"""Compare incumbent and alpha probabilities on out-of-sample observations.

The simulator is intentionally conservative: it models fixed target/stop outcomes,
optional slippage, and only enters when probability exceeds a configurable threshold.
It is a research tool, not a promise of future returns.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def simulate(d, pcol, threshold=.60, cost_bps=10.0):
    x=d[d[pcol]>=threshold].copy()
    if x.empty:return {'trades':0,'win_rate':0.,'expectancy_r':0.,'profit_factor':0.,'max_drawdown_r':0.,'total_r':0.}
    # target-before-stop label: +2R for target-first, -1R for stop-first, 0 for time/other.
    r=np.where(x.target_before_stop.astype(int).eq(1),2.0,-1.0)
    # Approximate round-trip execution cost in R using 1R=4% stop distance.
    r=r-(cost_bps/10000.0)/.04
    equity=np.cumsum(r); peak=np.maximum.accumulate(np.r_[0.,equity]); dd=equity-peak[1:]
    wins=r[r>0].sum(); losses=-r[r<0].sum()
    return {'trades':int(len(r)),'win_rate':float((r>0).mean()),'expectancy_r':float(r.mean()),'profit_factor':float(wins/losses) if losses else float('inf'),'max_drawdown_r':float(dd.min()),'total_r':float(r.sum())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='reports/alpha_oos_predictions.csv'); ap.add_argument('--output',default='reports/strategy_comparison.csv'); ap.add_argument('--thresholds',default='0.55,0.60,0.65,0.70,0.75'); ap.add_argument('--cost-bps',type=float,default=10.0); a=ap.parse_args()
    d=pd.read_csv(a.input); rows=[]
    # Benchmark: unconditional baseline at all observations; alpha strategies at thresholds.
    base=d.copy(); base['base_prob']=base.target_before_stop.mean()
    for name,pcol,t in [('BASE', 'base_prob',0.0)]+[(f'ALPHA_{t:.2f}','probability',t) for t in map(float,a.thresholds.split(','))]:
        m=simulate(base,pcol,t,a.cost_bps); m['strategy']=name; m['threshold']=t; rows.append(m)
    out=pd.DataFrame(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
