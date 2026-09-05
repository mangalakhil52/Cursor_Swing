#!/usr/bin/env python3
"""Run the lightweight research path on a bounded symbol universe."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--universe', default='data/nifty500.csv')
    p.add_argument('--history', default='data/history')
    p.add_argument('--symbols', type=int, default=100)
    p.add_argument('--output-dir', default='reports/fast')
    a=p.parse_args()
    u=pd.read_csv(a.universe)
    col=next((c for c in u.columns if c.lower() in {'symbol','ticker'}), None)
    if col is None: raise SystemExit(f'No symbol column in {a.universe}: {list(u.columns)}')
    u=u.dropna(subset=[col]).drop_duplicates(col).sort_values(col).head(a.symbols)
    out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    u.to_csv(out/'research_universe.csv', index=False)
    # build_alpha_dataset accepts a history directory, so temporarily expose only
    # the selected cached symbols through a lightweight staging directory.
    staging=out/'history'
    staging.mkdir(exist_ok=True)
    selected=set(u[col].astype(str))
    for src in Path(a.history).glob('*.csv'):
        if src.name == '_benchmark.csv' or src.stem in selected or src.stem.replace('.NS','') in selected:
            dst=staging/src.name
            if not dst.exists(): dst.write_bytes(src.read_bytes())
    bench=Path(a.history)/'_benchmark.csv'
    if bench.exists() and not (staging/'_benchmark.csv').exists(): (staging/'_benchmark.csv').write_bytes(bench.read_bytes())
    print(f'fast universe={len(u)} staged_history_files={len(list(staging.glob("*.csv")))}')

if __name__ == '__main__': main()
