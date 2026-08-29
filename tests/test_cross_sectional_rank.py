import pandas as pd
from src.cross_sectional_rank import rank_candidates,rank_by_date

def test_ranking_orders_candidates():
    d=pd.DataFrame({'symbol':['A','B'],'probability':[.9,.6],'structural_score':[90,70],'residual_momentum':[.8,.2],'relative_strength':[.9,.3],'volatility_efficiency':[.8,.2],'regime_fit':[.9,.3]})
    x=rank_candidates(d,top_k=1); assert x.iloc[0].symbol=='A' and int(x.selected.sum())==1

def test_rank_by_date():
    d=pd.DataFrame({'date':pd.to_datetime(['2026-01-01']*2),'symbol':['A','B'],'probability':[.8,.7]})
    x=rank_by_date(d,top_k=1); assert int(x.selected.sum())==1
