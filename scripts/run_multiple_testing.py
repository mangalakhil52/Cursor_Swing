#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from src.multiple_testing import benjamini_hochberg,block_bootstrap_mean

p=argparse.ArgumentParser(); p.add_argument('--input',default='reports/residual_alpha_oos.csv'); p.add_argument('--output',default='reports/multiple_testing.csv'); a=p.parse_args()
d=pd.read_csv(a.input); r=pd.to_numeric(d.get('residual_alpha',d.get('r_multiple',0)),errors='coerce').dropna(); boot=block_bootstrap_mean(r)
# Candidate family p-values may be supplied by upstream experiments; absent values are reported explicitly.
pvals=pd.to_numeric(d['model_p_value'],errors='coerce').dropna().tolist() if 'model_p_value' in d else []
bh=benjamini_hochberg(pvals) if pvals else {'threshold':None,'discoveries':0,'fdr_level':.10}
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame([{**boot,**{f'bh_{k}':v for k,v in bh.items()}}]).to_csv(out,index=False); print({**boot,**bh})
