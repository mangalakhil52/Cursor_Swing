from src.dynamic_sizing_engine import size_position

def test_high_vol_reduces_size():
    a=size_position(capital=1e6,entry=100,stop=95,probability=.8,vol_annual=.2)
    b=size_position(capital=1e6,entry=100,stop=95,probability=.8,vol_annual=.6)
    assert b['quantity']<a['quantity']

def test_drawdown_halt():
    x=size_position(capital=1e6,entry=100,stop=95,probability=.8,drawdown_pct=12)
    assert x['quantity']==0

def test_correlation_penalty():
    a=size_position(capital=1e6,entry=100,stop=95,probability=.8,correlation_penalty=1)
    b=size_position(capital=1e6,entry=100,stop=95,probability=.8,correlation_penalty=.5)
    assert b['quantity']<a['quantity']
