"""Final expected-edge gate for candidate signals."""
from __future__ import annotations

def evaluate(*, score:float, edge_probability:float, avg_win_r:float, avg_loss_r:float, min_score:float=68., min_probability:float=.60, min_rr:float=1.7)->dict:
    win=max(float(avg_win_r),0.0); loss=abs(min(float(avg_loss_r),0.0)); rr=win/loss if loss else float('inf')
    edge=float(edge_probability)*win-(1-float(edge_probability))*loss
    passed=score>=min_score and edge_probability>=min_probability and rr>=min_rr and edge>0
    return {'pass':bool(passed),'edge_r':edge,'reward_risk':rr,'reason':'positive expected R and minimum quality thresholds passed' if passed else 'reject: insufficient expected edge or quality'}
