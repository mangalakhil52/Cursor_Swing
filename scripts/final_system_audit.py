#!/usr/bin/env python3
"""Audit the research pipeline without confusing zero trades with a passing strategy."""
from pathlib import Path
import argparse
import json
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--candidates',default='reports/portfolio_candidates.csv'); p.add_argument('--executions',default='reports/execution_portfolio.csv'); p.add_argument('--costs',default='reports/execution_costs.csv'); p.add_argument('--output',default='reports/final_system_audit.json'); a=p.parse_args()
checks={}; reasons=[]
try:
 c=pd.read_csv(a.candidates); e=pd.read_csv(a.executions); k=pd.read_csv(a.costs)
 checks.update(candidate_file=True,execution_file=True,cost_file=True)
 direction=c['direction'].astype(str).str.upper() if 'direction' in c.columns else pd.Series(dtype=str)
 shorts=c.loc[direction.eq('SHORT')] if not c.empty else c
 checks['shorts_fno_gated']=bool(shorts.empty or ('fo_eligible' in shorts.columns and shorts.fo_eligible.fillna(False).astype(bool).all()))
 if not checks['shorts_fno_gated']: reasons.append('SHORT_SIGNAL_WITHOUT_FO_ELIGIBILITY')
 checks['candidate_duplicates']=bool(c.empty or not c.duplicated(['entry_date','symbol']).any())
 if not checks['candidate_duplicates']: reasons.append('DUPLICATE_DATE_SYMBOL_CANDIDATES')
 checks['execution_costs_present']=bool(k.empty or 'total_cost_bps' in k.columns)
 checks['net_r_present']=bool(e.empty or 'net_r' in e.columns)
 if not checks['net_r_present']: reasons.append('NET_R_MISSING')
 checks['execution_dates_valid']=True if e.empty or 'exit_date' not in e else bool(pd.to_datetime(e.exit_date,errors='coerce').ge(pd.to_datetime(e.entry_date,errors='coerce')).all())
 checks['positive_execution_coverage']=bool(len(e)>0)
 if not checks['positive_execution_coverage']: reasons.append('NO_EXECUTED_TRADES')
 if 'net_r' in e:
  nr=pd.to_numeric(e.net_r,errors='coerce').dropna(); checks['mean_net_r']=float(nr.mean()) if len(nr) else None; checks['win_rate']=float((nr>0).mean()) if len(nr) else None; checks['trades']=int(len(nr))
 else: checks['mean_net_r']=None; checks['win_rate']=None; checks['trades']=0
 checks['promotion_ready']=bool(all(checks[x] for x in ['candidate_file','execution_file','cost_file','shorts_fno_gated','candidate_duplicates','execution_costs_present','net_r_present','execution_dates_valid','positive_execution_coverage']))
except Exception as ex:
 checks['promotion_ready']=False; reasons.append('AUDIT_ERROR:'+str(ex))
checks['reasons']=reasons
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(checks,indent=2,default=str)); print(json.dumps(checks,indent=2)); raise SystemExit(0)
