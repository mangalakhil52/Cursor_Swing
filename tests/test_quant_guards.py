import pandas as pd
from src.edge_gate import evaluate
from src.regime_gate import allow_direction
from src.signal_dedup import deduplicate_signals
from src.validation_suite import validate_results

def test_edge_gate_rejects_negative_expected_value():
    r=evaluate(score=80,edge_probability=.61,avg_win_r=1,avg_loss_r=-2)
    assert not r['pass'] and r['edge_r']<0

def test_regime_gate_is_conservative():
    assert allow_direction('SHORT','BULL') is False
    assert allow_direction('LONG','BEAR') is False

def test_signal_dedup():
    d=pd.DataFrame({'execution_date':['2026-08-01','2026-08-02','2026-08-06'],'symbol':['ABC']*3,'direction':['LONG']*3,'score':[90,80,85]})
    out=deduplicate_signals(d,3)
    assert len(out)==2 and out.iloc[0].score==90

def test_validation_rejects_temporal_error():
    d=pd.DataFrame({'signal_date':['2026-08-03'],'execution_date':['2026-08-02'],'exit_date':['2026-08-04'],'r_multiple':[1.0],'direction':['LONG']})
    assert validate_results(d)['passed'] is False
