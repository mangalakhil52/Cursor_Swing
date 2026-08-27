"""Regime-aware entry policy. The model may abstain when direction conflicts with regime."""
from __future__ import annotations

def allow_direction(direction:str, regime:str, *, long_in_bear=False, short_in_bull=False)->bool:
    d=str(direction).upper(); r=str(regime).upper()
    if d=='SHORT' and r in {'STRONG_BULL','BULL'}: return bool(short_in_bull)
    if d=='LONG' and r in {'STRONG_BEAR','BEAR'}: return bool(long_in_bear)
    return d in {'LONG','SHORT'}
