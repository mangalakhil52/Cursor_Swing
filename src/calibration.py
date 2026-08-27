"""Out-of-sample probability calibration and ranking diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd


def isotonic_like_bins(y: pd.Series, score: pd.Series, bins: int = 10) -> pd.DataFrame:
    x=pd.DataFrame({"score":score.astype(float),"y":y.astype(float)}).dropna()
    if x.empty:return pd.DataFrame(columns=["bucket","n","score_mean","event_rate"])
    x["bucket"]=pd.qcut(x.score.rank(method="first"),min(bins,len(x)),labels=False)+1
    return x.groupby("bucket",observed=True).agg(n=("y","size"),score_mean=("score","mean"),event_rate=("y","mean")).reset_index()


def brier_score(y: pd.Series, p: pd.Series) -> float:
    x=pd.concat([y,p],axis=1).dropna();
    return float(((x.iloc[:,0]-x.iloc[:,1])**2).mean()) if len(x) else float("nan")


def calibration_gap(y: pd.Series, p: pd.Series, bins: int = 10) -> float:
    x=pd.concat([y,p],axis=1).dropna()
    if x.empty:return float("nan")
    x.columns=["y","p"]; x["b"]=pd.qcut(x.p.rank(method="first"),min(bins,len(x)),labels=False)
    g=x.groupby("b",observed=True).agg(y=("y","mean"),p=("p","mean"),n=("y","size"))
    return float((g.n*(g.y-g.p).abs()).sum()/g.n.sum())


def monotonicity(report: pd.DataFrame) -> float:
    if report.empty or len(report)<2:return float("nan")
    return float(report["avg_r"].corr(report["score_decile"],method="spearman"))
