"""Guardrails for fast research mode.

Fast mode is an iteration harness only; full Nifty 500 validation remains mandatory
before promoting strategy changes.
"""
from __future__ import annotations
import argparse
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--max-mb',type=float,default=250); a=p.parse_args()
f=Path(a.dataset)
if not f.exists(): raise SystemExit(f'Missing dataset: {f}')
size=f.stat().st_size/1024/1024
print(f'fast_dataset_mb={size:.1f}')
if size>a.max_mb: raise SystemExit('Fast dataset exceeds configured size guard; reduce research universe.')
