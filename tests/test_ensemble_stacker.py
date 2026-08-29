import pandas as pd
from src.ensemble_stacker import expanding_stack,accept_stack

def test_stack_is_oos_only():
    n=240; d=pd.DataFrame({'probability':[.6,.8]*(n//2),'meta_probability':[.55,.75]*(n//2),'ensemble_probability':[.58,.78]*(n//2),'cross_sectional_rank':[1,2]*(n//2),'target_before_stop':[0,1]*(n//2)})
    x=expanding_stack(d,min_train=200,step=20); assert x.stack_probability.iloc[:200].isna().all(); assert x.stack_probability.iloc[200:].notna().any()

def test_disagreement_gate():
    d=pd.DataFrame({'stack_probability':[.7,.7],'model_disagreement':[.05,.2]}); x=accept_stack(d); assert list(x)==[True,False]
