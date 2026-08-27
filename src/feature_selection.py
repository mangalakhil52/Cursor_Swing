"""Simple, deterministic feature-ablation diagnostics for research."""
from __future__ import annotations
import pandas as pd

GROUPS={
    "momentum":["mom3","mom5","mom10","mom20","mom60"],
    "trend":["efficiency"],
    "volatility":["ann_vol"],
    "flow":["volume_ratio"],
    "residual_alpha":["residual"],
    "baseline_score":["score"],
}

def available_groups(df: pd.DataFrame)->dict[str,list[str]]:
    return {k:[c for c in cols if c in df.columns] for k,cols in GROUPS.items() if any(c in df.columns for c in cols)}
