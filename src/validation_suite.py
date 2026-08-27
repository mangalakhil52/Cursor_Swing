"""Leakage and backtest sanity checks used before accepting research output."""
from __future__ import annotations
import pandas as pd


def validate_results(results: pd.DataFrame) -> dict:
    checks={}
    if results.empty:return {'passed':False,'checks':{'non_empty':False},'errors':['no research rows']}
    required={'signal_date','execution_date','exit_date','r_multiple','direction'}
    missing=required-set(results.columns); checks['required_columns']=not missing
    dates=pd.to_datetime(results['signal_date']); exec_dates=pd.to_datetime(results['execution_date']); exits=pd.to_datetime(results['exit_date'])
    checks['execution_after_signal']=bool((exec_dates>dates).all())
    checks['exit_after_execution']=bool((exits>=exec_dates).all())
    checks['no_future_signal_dates']=bool((dates<=exec_dates).all())
    checks['finite_r']=bool(pd.to_numeric(results.r_multiple,errors='coerce').notna().all())
    checks['valid_direction']=bool(results.direction.astype(str).str.upper().isin(['LONG','SHORT']).all())
    errors=[k for k,v in checks.items() if not v]
    return {'passed':not errors,'checks':checks,'errors':errors}
