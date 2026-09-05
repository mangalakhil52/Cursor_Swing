#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

required=['alpha_oos_predictions.csv','alpha_oos_enriched.csv','cross_sectional_rank.csv','alpha_ensemble_oos.csv','meta_label_oos.csv','ensemble_stack_oos.csv','residual_alpha_oos.csv']
root=Path('reports/fast')
missing=[x for x in required if not (root/x).exists()]
if missing: raise SystemExit('Missing fast artifacts: '+', '.join(missing))
for name in required:
    d=pd.read_csv(root/name)
    if d.empty: raise SystemExit(f'Empty artifact: {name}')
    print(name, len(d), 'rows')
print('FAST_SMOKE_OK')
