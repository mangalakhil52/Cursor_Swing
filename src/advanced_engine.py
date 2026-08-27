"""Advanced quantitative weekly-swing research engine.

This is a probabilistic ranking layer, not a certainty machine. It combines
multiple independent-ish measurements using only information available at the
signal close: volatility-normalised multi-horizon momentum, trend efficiency,
persistence, return-path entropy, volume surprise, signed-flow proxy,
structure, beta-adjusted residual strength, volatility regime and tail risk.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class AdvancedSignal:
    score: float; edge_probability: float; regime_score: float; momentum_score: float; trend_score: float; flow_score: float; structure_score: float; risk_score: float; expected_r_multiple: float; volatility_regime: str; persistence: float; entropy: float; efficiency_ratio: float; residual_strength: float; reasons: tuple[str,...]; reject_reasons: tuple[str,...]

def _ret(c,n):
    if len(c)<=n or float(c.iloc[-n-1])==0:return 0.
    return float(c.iloc[-1]/c.iloc[-n-1]-1.)

def _z(x,s):
    s=s.dropna().astype(float)
    if len(s)<10:return 0.
    return (x-float(s.mean()))/max(float(s.std(ddof=1)),1e-9)

def _entropy(r):
    r=r.dropna().astype(float)
    if len(r)<20:return 1.
    q=r.quantile([.25,.5,.75]).to_numpy(); states=np.digitize(r.to_numpy(),q,right=True); counts=np.bincount(states,minlength=4).astype(float); p=counts[counts>0]/counts.sum()
    return float(-(p*np.log2(p)).sum()/2.)

def _hurst(r):
    r=r.dropna().astype(float).to_numpy()
    if len(r)<30:return .5
    vals=[]
    for n in [8,12,16,24,min(32,len(r))]:
        x=r[-n:]; y=x-x.mean(); sd=y.std(ddof=1)
        if sd<=0:continue
        rs=(np.cumsum(y).max()-np.cumsum(y).min())/sd
        if rs>0:vals.append((math.log(n),math.log(rs)))
    return float(np.polyfit(np.array(vals)[:,0],np.array(vals)[:,1],1)[0]) if len(vals)>=2 else .5

def _eff(c,n=20):
    if len(c)<=n:return 0.
    return float(abs(c.iloc[-1]-c.iloc[-n-1])/max(float(c.diff().abs().tail(n).sum()),1e-12))

def _residual(stock,bench,n=60):
    if bench is None or len(stock)<n or len(bench)<n:return 0.
    s=stock.pct_change().tail(n).dropna(); b=bench.pct_change().tail(n).dropna(); j=pd.concat([s.rename('s'),b.rename('b')],axis=1).dropna()
    if len(j)<30:return 0.
    beta=np.cov(j.s,j.b,ddof=1)[0,1]/max(float(np.var(j.b,ddof=1)),1e-12); return float((j.s-beta*j.b).tail(20).sum()*100.)

def compute_advanced(daily:pd.DataFrame,benchmark:pd.Series|None,direction:str)->AdvancedSignal:
    c=daily.close.astype(float); h=daily.high.astype(float); l=daily.low.astype(float); v=daily.volume.astype(float); r=c.pct_change(); atr=(h-l).rolling(14).mean(); atr_pct=float(atr.iloc[-1]/c.iloc[-1]*100) if c.iloc[-1] else 0.
    sign=1. if direction=="LONG" else -1.; m=np.array([_ret(c,n) for n in (3,5,10,20,60)]); sd20=float(r.rolling(20).std().iloc[-1]); ann=sd20*math.sqrt(252) if pd.notna(sd20) else 0.; mom=sign*float(np.dot(m,[.08,.14,.22,.31,.25])); mom_score=float(np.tanh(mom/max(ann,.03))*50+50)
    er=_eff(c); hurst=_hurst(r.tail(80)); ent=_entropy(r.tail(60)); persistence=max(0.,min(1.,.55*er+.45*max(0.,min(1.,(hurst-.35)/.45)))); trend=100*(.60*er+.40*persistence)
    vm=v.rolling(20).mean(); vs=float(v.iloc[-1]/max(float(vm.iloc[-1]),1.)); signed=float((np.sign(r.tail(20))*v.tail(20)).sum()/max(float(v.tail(20).sum()),1.)); flow=max(0.,min(100.,50+28*math.tanh((vs-1)/.7)+22*sign*signed))
    hi=float(h.iloc[:-1].tail(20).max()); lo=float(l.iloc[:-1].tail(20).min()); loc=((float(c.iloc[-1])-lo)/(hi-lo) if direction=="LONG" else (hi-float(c.iloc[-1]))/(hi-lo)) if hi!=lo else .5; br=((float(c.iloc[-1])-hi)/max(float(atr.iloc[-1]),1e-9) if direction=="LONG" else (lo-float(c.iloc[-1]))/max(float(atr.iloc[-1]),1e-9)); structure=max(0.,min(100.,45+35*(loc-.5)+20*math.tanh(br)))
    residual=sign*_residual(c,benchmark); residual_score=50+50*math.tanh(residual/3.)
    ah=(atr/c*100).dropna().tail(100); az=_z(atr_pct,ah)
    if az>2 or atr_pct>8:reg="EXTREME"; risk=25.
    elif az<-1.5 or atr_pct<1.2:reg="COMPRESSED"; risk=35.
    elif abs(az)<=1.25:reg="NORMAL"; risk=85.
    else:reg="EXPANDING"; risk=72.
    extension=abs(float(c.iloc[-1]/c.ewm(span=21,adjust=False).mean().iloc[-1]-1.)); dd20=float(c.iloc[-1]/c.tail(20).max()-1.)
    if extension>.12:risk-=30
    if direction=="LONG" and dd20<-.08:risk-=10
    if direction=="SHORT" and dd20>-.02:risk-=5
    risk=max(0.,min(100.,risk)); regime=50+(20 if persistence>.65 and ent<.85 else 0)+(15 if reg in ("NORMAL","EXPANDING") else 0)+(15 if residual_score>65 else 0); regime=max(0.,min(100.,regime))
    score=.23*mom_score+.20*trend+.16*flow+.14*structure+.12*residual_score+.10*regime+.05*risk; edge=.50+.47*math.tanh((score-68)/13); expected=max(-.5,min(3.5,(edge-.5)*5))
    reasons=[]; rejects=[]
    if persistence>=.65:reasons.append(f"persistent trend path (ER {er:.2f}, Hurst {hurst:.2f})")
    if residual_score>=65:reasons.append(f"positive beta-adjusted residual strength ({residual:.2f}%)")
    if flow>=70:reasons.append(f"volume/flow expansion ({vs:.2f}x, signed flow {signed:+.2f})")
    if structure>=70:reasons.append("strong price-location / breakout structure")
    if ent<=.70:reasons.append(f"low return-path entropy ({ent:.2f})")
    if reg=="EXTREME":rejects.append("extreme volatility regime")
    if reg=="COMPRESSED":rejects.append("compressed volatility regime")
    if extension>.12:rejects.append("price extension >12% from 21 EMA")
    if risk<35:rejects.append("poor volatility/tail-risk profile")
    if er<.30:rejects.append(f"weak directional efficiency ({er:.2f})")
    if persistence<.40:rejects.append(f"weak persistence ({persistence:.2f})")
    if ent>.92:rejects.append(f"high return-path entropy ({ent:.2f})")
    if score<68:rejects.append(f"advanced composite below threshold ({score:.1f})")
    return AdvancedSignal(round(score,2),round(edge,4),round(regime,2),round(mom_score,2),round(trend,2),round(flow,2),round(structure,2),round(risk,2),round(expected,2),reg,round(persistence,3),round(ent,3),round(er,3),round(residual,3),tuple(reasons),tuple(rejects))
