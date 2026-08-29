from src.execution_costs import estimate_cost_bps,liquidity_gate

def test_impact_increases_with_participation():
    a=estimate_cost_bps(price=100,quantity=100,adv_value=1_000_000); b=estimate_cost_bps(price=100,quantity=10_000,adv_value=1_000_000); assert b['total_cost_bps']>a['total_cost_bps']

def test_liquidity_gate():
    assert liquidity_gate(participation=.03); assert not liquidity_gate(participation=.08)
