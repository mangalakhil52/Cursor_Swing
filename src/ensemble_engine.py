"""Regime-aware ensemble score for research/shadow mode.

Combines the existing structural score with calibrated OOS alpha probability. This module
is deliberately isolated from production until OOS comparison proves incremental value.
"""
from __future__ import annotations

REGIME_WEIGHT={
    'STRONG_BULL': (0.35,0.65),'BULL':(0.40,0.60),'SIDEWAYS':(0.60,0.40),
    'HIGH_VOL_SIDEWAYS':(0.70,0.30),'BEAR':(0.55,0.45),'STRONG_BEAR':(0.35,0.65)
}

def ensemble_score(structural_score:float, alpha_probability:float, regime:str)->float:
    sw,aw=REGIME_WEIGHT.get(str(regime).upper(),(0.60,0.40))
    return 100.0*(sw*max(0,min(100,float(structural_score)))/100.0 + aw*max(0,min(1,float(alpha_probability))))

def decision(structural_score:float, alpha_probability:float, regime:str, *, threshold:float=68.0, probability_floor:float=.60)->dict:
    score=ensemble_score(structural_score,alpha_probability,regime)
    passed=score>=threshold and alpha_probability>=probability_floor
    return {'ensemble_score':score,'pass':passed,'regime':str(regime).upper(),'reason':'ensemble and alpha probability thresholds passed' if passed else 'reject: ensemble confidence below gate'}
