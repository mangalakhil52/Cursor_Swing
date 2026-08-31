"""Regime-aware ensemble score for research/shadow mode."""
from __future__ import annotations

# (structural weight, alpha weight).  Bull regimes deliberately favour
# alpha evidence; sideways regimes favour structural evidence.
REGIME_WEIGHT={
    'STRONG_BULL': (0.30,0.70),'BULL':(0.35,0.65),'SIDEWAYS':(0.55,0.45),
    'HIGH_VOL_SIDEWAYS':(0.65,0.35),'BEAR':(0.50,0.50),'STRONG_BEAR':(0.60,0.40)
}


def ensemble_score(structural_score:float, alpha_probability:float, regime:str)->float:
    """Return a 0-100 regime-weighted ensemble score.

    A weighted ensemble only changes the result when structural and alpha
    evidence differ.  This is intentional: identical 80/80 inputs must
    produce the same score regardless of regime because there is nothing for
    the regime weighting to prefer.
    """
    sw,aw=REGIME_WEIGHT.get(str(regime).upper(),(0.55,0.45))
    structural=max(0.0,min(100.0,float(structural_score)))
    alpha=max(0.0,min(1.0,float(alpha_probability)))*100.0
    return sw*structural + aw*alpha


def decision(structural_score:float, alpha_probability:float, regime:str, *, threshold:float=68.0, probability_floor=.60)->dict:
    score=ensemble_score(structural_score,alpha_probability,regime)
    passed=score>=threshold and float(alpha_probability)>=probability_floor
    return {'ensemble_score':score,'pass':passed,'regime':str(regime).upper(),'reason':'ensemble and alpha probability thresholds passed' if passed else 'reject: ensemble confidence below gate'}
