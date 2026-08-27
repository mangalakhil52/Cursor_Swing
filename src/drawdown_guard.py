"""Live-signal drawdown guardrails; defensive, not predictive."""
from __future__ import annotations

def allowed_after_drawdown(current_drawdown_pct:float, soft_limit_pct:float=8.0, hard_limit_pct:float=12.0)->dict:
    dd=abs(float(current_drawdown_pct))
    if dd>=hard_limit_pct:return {'allowed':False,'risk_multiplier':0.0,'state':'HALT'}
    if dd>=soft_limit_pct:return {'allowed':True,'risk_multiplier':0.5,'state':'DEFENSIVE'}
    return {'allowed':True,'risk_multiplier':1.0,'state':'NORMAL'}
