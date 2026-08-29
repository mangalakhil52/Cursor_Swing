import pandas as pd
from src.portfolio_engine import simulate,PortfolioConfig

def test_concurrent_cap_is_enforced():
    d=pd.DataFrame({'symbol':['A','B','C'],'entry_date':['2026-01-01']*3,'exit_date':['2026-01-10']*3,'score':[90,80,70],'entry_price':[100,100,100],'stop_price':[95,95,95],'r_multiple':[2,2,2]})
    a,r=simulate(d,PortfolioConfig(capital=100000,max_positions=2))
    assert len(a)==2 and len(r)==1 and r.iloc[0].rejection_reason=='MAX_CONCURRENT_POSITIONS'
