#!/usr/bin/env python3
"""Join point-in-time feature columns onto OOS predictions without changing labels."""
from pathlib import Path
import argparse
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--predictions',default='reports/alpha_oos_predictions.csv')
p.add_argument('--dataset',default='reports/alpha_dataset.parquet')
p.add_argument('--output',default='reports/alpha_oos_enriched.csv')
a=p.parse_args()

pred=pd.read_csv(a.predictions)
feat=pd.read_parquet(a.dataset)
pred['date']=pd.to_datetime(pred['date'])
feat['date']=pd.to_datetime(feat['date'])
key=['date','symbol']
feature_cols=[c for c in feat.columns if c not in {'target_before_stop','forward_return_5','forward_return_10','mfe_10','mae_10','triple_barrier'}]
feat=feat[feature_cols].drop_duplicates(key)
merged=pred.merge(feat,on=key,how='left',suffixes=('','_dataset'))
# Prediction-time values remain authoritative for labels/probabilities.
for c in list(merged.columns):
    if c.endswith('_dataset'):
        base=c[:-8]
        if base not in merged or merged[base].isna().all():
            merged[base]=merged[c]
        merged.drop(columns=[c],inplace=True)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
merged.to_csv(a.output,index=False)
print(f'rows={len(merged)} matched_features={int(merged["ret_20"].notna().sum()) if "ret_20" in merged else 0}')
