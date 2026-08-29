import numpy as np
from src.portfolio_risk import monte_carlo_risk, max_drawdown

def test_drawdown_is_non_positive():
    assert max_drawdown(np.array([1.,-2.,1.])) <= 0

def test_monte_carlo_outputs_probabilities():
    r=np.array([2.,-1.,2.,-1.,2.,-1.,2.,-1.,2.,-1.])
    x=monte_carlo_risk(r,simulations=100,horizon=20,block=2)
    assert 0<=x['ruin_probability']<=1
    assert x['simulations']==100
