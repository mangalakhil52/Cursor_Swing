"""Automated robustness gate for research reports."""
from __future__ import annotations
import pandas as pd


def gate(time_blocks:pd.DataFrame, stress_summary:dict, *, min_blocks=5, min_positive_blocks=.60)->dict:
    reasons=[]
    if len(time_blocks)<min_blocks: reasons.append('insufficient independent time blocks')
    if not time_blocks.empty and float((time_blocks.avg_r>0).mean())<min_positive_blocks: reasons.append('positive expectancy is not stable across time blocks')
    if not stress_summary.get('stable',False): reasons.append('nearby parameter perturbations are unstable')
    return {'pass':not reasons,'reasons':reasons or ['robustness criteria passed']}
