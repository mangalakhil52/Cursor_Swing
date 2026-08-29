import pandas as pd
from src.exposure_engine import approve

def test_sector_concentration():
    ok,reason=approve({'symbol':'B','sector':'BANK','beta':1},[{'symbol':'A','sector':'BANK'}])
    assert not ok and reason=='SECTOR_CONCENTRATION'

def test_correlation_cluster():
    c=pd.DataFrame([[1,.9],[.9,1]],index=['A','B'],columns=['A','B'])
    ok,reason=approve({'symbol':'B','sector':'TECH','beta':1},[{'symbol':'A','sector':'BANK'}],c)
    assert not ok and reason=='CORRELATION_CLUSTER'

def test_beta_limit():
    ok,reason=approve({'symbol':'B','sector':'TECH','beta':1.8},[])
    assert not ok and reason=='MARKET_BETA_LIMIT'
