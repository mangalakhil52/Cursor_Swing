import numpy as np
from src.uncertainty import uncertainty,confidence_gate

def test_uncertainty_metrics():
    x=uncertainty([0,1,1,0],[.1,.9,.8,.2])
    assert 0<=x['brier']<=1
    assert x['coverage']==1

def test_confidence_gate():
    assert confidence_gate(.70,.05)
    assert not confidence_gate(.70,.20)
