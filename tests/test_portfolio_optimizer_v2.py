import pandas as pd
from src.portfolio_optimizer import select_diversified

def test_sector_cap_limits_concentration():
    c=pd.DataFrame({'symbol':['A','B','C'],'ml_probability':[.9,.89,.88],'score':[90,89,88],'sector':['IT','IT','BANK'],'vol_annual':[.2,.2,.2]})
    r=pd.DataFrame({'A':[.01,.02,.01],'B':[.01,.02,.01],'C':[.00,.01,.02]})
    x=select_diversified(c,r,max_picks=3,max_sector_weight=.40)
    assert (x.sector=='IT').sum()<=2
