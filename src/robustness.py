"""Robustness diagnostics for swing research results.

These tests are descriptive only; they do not optimize parameters on the test set.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def expectancy_by_regime(results: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
    if results.empty or regime_col not in results.columns:
        return pd.DataFrame()
    return results.groupby(regime_col, observed=True).agg(
        trades=("r_multiple", "size"),
        win_rate=("r_multiple", lambda s: float((s > 0).mean())),
        avg_r=("r_multiple", "mean"),
        median_r=("r_multiple", "median"),
        profit_factor=("r_multiple", lambda s: float(s[s > 0].sum() / abs(s[s < 0].sum())) if (s < 0).any() else np.inf),
    ).reset_index()


def symbol_concentration(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:return pd.DataFrame()
    g=results.groupby("symbol",observed=True).agg(trades=("r_multiple","size"),cum_r=("r_multiple","sum"),avg_r=("r_multiple","mean")).reset_index()
    total=float(g.cum_r.sum())
    g["share_of_net_r"]=g.cum_r/total if total else 0.
    return g.sort_values("cum_r",ascending=False)


def block_stability(results: pd.DataFrame, blocks: int = 5) -> pd.DataFrame:
    if results.empty:return pd.DataFrame()
    x=results.sort_values("execution_date").copy(); x["block"]=pd.qcut(np.arange(len(x)),min(blocks,len(x)),labels=False,duplicates="drop")+1
    return x.groupby("block",observed=True).agg(trades=("r_multiple","size"),avg_r=("r_multiple","mean"),win_rate=("r_multiple",lambda s:(s>0).mean()),cum_r=("r_multiple","sum")).reset_index()
