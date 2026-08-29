from src.position_sizing import size,fractional_kelly

def test_kelly_never_negative(): assert fractional_kelly(.2)>=0

def test_drawdown_halt(): assert size(100,95,.8,drawdown_pct=12)['quantity']==0

def test_high_vol_reduces_risk():
    normal=size(100,95,.8,realized_vol_annual=.20)['risk_pct']
    high=size(100,95,.8,realized_vol_annual=.60)['risk_pct']
    assert high<normal

def test_exposure_cap():
    x=size(100,95,.99)
    assert x['notional']<=500000
