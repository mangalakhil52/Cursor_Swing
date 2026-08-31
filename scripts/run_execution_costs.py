#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.execution_costs import estimate_cost_bps,liquidity_gate

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/execution_portfolio.csv'); p.add_argument('--output',default='reports/execution_costs.csv'); a=p.parse_args()
d=pd.read_csv(a.input); rows=[]
for _,r in d.iterrows():
    adv=r.get('adv_value',r.get('average_daily_value',None)); q=r.get('quantity',0); price=r.get('entry_price',r.get('entry',0)); z=estimate_cost_bps(price=float(price),quantity=float(q),adv_value=float(adv) if pd.notna(adv) else None); z.update({'symbol':r.get('symbol'),'liquidity_ok':liquidity_gate(participation=z['participation'])}); rows.append(z)
cols=['notional','participation','spread_cost_bps','slippage_bps','impact_bps','total_cost_bps','symbol','liquidity_ok']; out=pd.DataFrame(rows,columns=cols); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f'rows={len(out)} liquidity_ok={int(out.liquidity_ok.sum()) if len(out) else 0}')
