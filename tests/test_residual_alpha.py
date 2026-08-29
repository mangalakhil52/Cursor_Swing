import pandas as pd
from src.residual_alpha import expanding_residual_alpha

def test_residual_is_future_blind():
    n=300; d=pd.DataFrame({'probability':[.6,.8]*(n//2),'stack_probability':[.55,.75]*(n//2),'target_before_stop':[0,1]*(n//2),'rsi':[40,70]*(n//2),'atr_pct':[.02,.03]*(n//2),'volume_ratio':[1.,2.]*(n//2),'relative_strength':[.2,.8]*(n//2),'residual_momentum':[.1,.7]*(n//2),'regime_score':[.5,.9]*(n//2),'distance_to_resistance':[.8,.2]*(n//2),'distance_to_support':[.2,.8]*(n//2),'structural_score':[60,80]*(n//2)})
    x=expanding_residual_alpha(d,min_train=250,step=25); assert x.residual_alpha.iloc[:250].isna().all(); assert x.residual_alpha.iloc[250:].notna().any()
