"""Conservative execution-cost model for swing backtests."""
from __future__ import annotations
import numpy as np

def estimate_cost_bps(*, price:float, quantity:float, adv_value:float|None=None, spread_bps:float=8., base_slippage_bps:float=5., impact_coeff_bps:float=35.) -> dict:
    notional=max(float(price)*abs(float(quantity)),0.)
    adv=max(float(adv_value or 0.),1.)
    participation=notional/adv
    # Square-root impact is a standard conservative market-impact approximation.
    impact=impact_coeff_bps*np.sqrt(max(participation,0.))
    total=spread_bps/2+base_slippage_bps+impact
    return {'notional':notional,'participation':participation,'spread_cost_bps':spread_bps/2,'slippage_bps':base_slippage_bps,'impact_bps':impact,'total_cost_bps':total}

def apply_cost(price:float,direction:str,total_cost_bps:float)->float:
    c=float(total_cost_bps)/10000.; p=float(price)
    return p*(1+c) if str(direction).upper()=='SHORT' else p*(1+c)

def liquidity_gate(*, participation:float,max_participation:float=.05)->bool:
    return bool(float(participation)<=max_participation)
