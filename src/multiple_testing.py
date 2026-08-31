"""Controls for selection bias when many hypotheses/models are tested."""
from __future__ import annotations
import numpy as np
import pandas as pd

def benjamini_hochberg(p_values, alpha=.10):
    p=np.asarray(p_values,dtype=float); p=p[np.isfinite(p)]; n=len(p)
    if n==0:return {'threshold':0.,'discoveries':0,'fdr_level':alpha}
    order=np.argsort(p); ranked=p[order]; cutoff=np.arange(1,n+1)*alpha/n; ok=ranked<=cutoff
    threshold=float(ranked[np.where(ok)[0].max()]) if ok.any() else 0.
    return {'threshold':threshold,'discoveries':int((p<=threshold).sum()),'fdr_level':alpha}

def block_bootstrap_mean(r, block=5, iterations=2000, seed=41):
    x=np.asarray(r,dtype=float); x=x[np.isfinite(x)]; n=len(x)
    if n==0 or n<block:return {'mean':float(np.mean(x)) if n else np.nan,'lo':np.nan,'hi':np.nan,'p_positive':np.nan}
    block=int(max(1,block)); rng=np.random.default_rng(seed); starts=np.arange(n-block+1); means=[]
    for _ in range(iterations):
        sample=[]
        while len(sample)<n:
            s=int(rng.choice(starts)); sample.extend(x[s:s+block])
        means.append(np.mean(sample[:n]))
    a=np.asarray(means); return {'mean':float(np.mean(x)),'lo':float(np.quantile(a,.025)),'hi':float(np.quantile(a,.975)),'p_positive':float(np.mean(a>0))}
