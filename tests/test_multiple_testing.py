from src.multiple_testing import benjamini_hochberg,block_bootstrap_mean

def test_bh_controls_discoveries():
    x=benjamini_hochberg([.001,.01,.2,.8],alpha=.10); assert x['discoveries']>=1

def test_bootstrap_has_interval():
    x=block_bootstrap_mean([1,-1,1,-1,2,-2]*10,block=3,iterations=100); assert x['lo']<=x['mean']<=x['hi']
