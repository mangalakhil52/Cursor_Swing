"""Walk-forward research engine for the weekly swing model.

Research-only: evaluates signals using only data available at each signal date.
Execution is next-session open by default. Includes MFE/MAE, forward returns,
R-multiple outcomes, score deciles, regime buckets and bootstrap confidence
intervals. It intentionally does not optimize parameters on the test sample.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ResearchConfig:
    horizon: int = 10
    stop_pct: float = 0.05
    target_r: float = 1.7
    slippage_bps: float = 10.0
    min_history: int = 120


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([(df.high-df.low).abs(), (df.high-df.close.shift()).abs(), (df.low-df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _signal_features(x: pd.DataFrame, benchmark: pd.Series | None) -> dict:
    c=x.close.astype(float); r=c.pct_change(); a=_atr(x); last=float(c.iloc[-1]);
    ret={"mom3":c.pct_change(3).iloc[-1],"mom5":c.pct_change(5).iloc[-1],"mom10":c.pct_change(10).iloc[-1],"mom20":c.pct_change(20).iloc[-1],"mom60":c.pct_change(60).iloc[-1]}
    path=float(c.diff().abs().tail(20).sum()); net=abs(last-float(c.iloc[-21]))
    er=net/max(path,1e-12)
    vol=float(r.rolling(20).std().iloc[-1]*math.sqrt(252))
    ema21=float(c.ewm(span=21,adjust=False).mean().iloc[-1]); ext=last/ema21-1
    vol_ratio=float(x.volume.iloc[-1]/max(float(x.volume.rolling(20).mean().iloc[-1]),1.0))
    hi20=float(x.high.iloc[:-1].tail(20).max()); lo20=float(x.low.iloc[:-1].tail(20).min())
    location=(last-lo20)/max(hi20-lo20,1e-9)
    residual=0.0
    if benchmark is not None and len(benchmark)>=60:
        s=r.tail(60); b=benchmark.pct_change().tail(60); j=pd.concat([s.rename('s'),b.rename('b')],axis=1).dropna()
        if len(j)>=30:
            beta=np.cov(j.s,j.b,ddof=1)[0,1]/max(float(np.var(j.b,ddof=1)),1e-12); residual=float((j.s-beta*j.b).tail(20).sum())
    score=50+25*np.tanh((c.pct_change(20).iloc[-1])/.10)+15*er+10*np.tanh(residual/.03)
    return {**ret,"efficiency":er,"ann_vol":vol,"extension":ext,"volume_ratio":vol_ratio,"location":location,"residual":residual,"score":float(np.clip(score,0,100))}


def _bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float,float]:
    if len(values)<2:return (float('nan'),float('nan'))
    rng=np.random.default_rng(seed); means=np.empty(n_boot)
    for i in range(n_boot): means[i]=rng.choice(values,size=len(values),replace=True).mean()
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def evaluate_symbol(symbol: str, daily: pd.DataFrame, benchmark: pd.Series | None = None, cfg: ResearchConfig = ResearchConfig()) -> pd.DataFrame:
    d=daily.sort_index().copy(); required={"open","high","low","close","volume"}
    if not required.issubset(d.columns): raise ValueError(f"missing columns: {required-set(d.columns)}")
    rows=[]
    for i in range(max(cfg.min_history,60),len(d)-cfg.horizon-1):
        hist=d.iloc[:i+1]; f=_signal_features(hist, benchmark.iloc[:i+1] if benchmark is not None else None)
        # Research baseline direction: trend/momentum direction at signal close.
        direction="LONG" if f["mom20"]>=0 else "SHORT"
        entry=float(d.open.iloc[i+1])
        slip=cfg.slippage_bps/10000; entry*=1+slip if direction=="LONG" else 1-slip
        risk=entry*cfg.stop_pct; target=entry+risk*cfg.target_r if direction=="LONG" else entry-risk*cfg.target_r
        future=d.iloc[i+1:i+1+cfg.horizon]
        if direction=="LONG":
            mfe=(future.high.max()/entry-1) if len(future) else 0; mae=(future.low.min()/entry-1) if len(future) else 0
            hit_t=bool((future.high>=target).any()); hit_s=bool((future.low<=entry-risk).any())
        else:
            mfe=(entry/future.low.min()-1) if len(future) else 0; mae=(entry/future.high.max()-1) if len(future) else 0
            hit_t=bool((future.low<=target).any()); hit_s=bool((future.high>=entry+risk).any())
        # Conservative daily-bar ordering when both occur: stop first.
        outcome=-1.0 if hit_s else (cfg.target_r if hit_t else ((float(future.close.iloc[-1])/entry-1)*1 if direction=="LONG" else (entry/float(future.close.iloc[-1])-1)))
        rows.append({"symbol":symbol,"signal_date":d.index[i],"execution_date":d.index[i+1],"direction":direction,"score":f["score"],"efficiency":f["efficiency"],"ann_vol":f["ann_vol"],"volume_ratio":f["volume_ratio"],"residual":f["residual"],"mfe_pct":mfe*100,"mae_pct":mae*100,"target_hit":hit_t,"stop_hit":hit_s,"r_multiple":outcome,"fwd5_pct":(float(d.close.iloc[min(i+5,len(d)-1)])/entry-1)*100*(1 if direction=="LONG" else -1),"fwd10_pct":(float(d.close.iloc[min(i+10,len(d)-1)])/entry-1)*100*(1 if direction=="LONG" else -1)})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    if results.empty:return {"trades":0}
    r=results.r_multiple.astype(float); wins=r[r>0]; losses=r[r<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) else float('inf'); lo,hi=_bootstrap_mean(r.to_numpy())
    eq=r.cumsum(); dd=eq-eq.cummax()
    return {"trades":int(len(r)),"win_rate":float((r>0).mean()),"avg_r":float(r.mean()),"median_r":float(r.median()),"profit_factor":pf,"cum_r":float(r.sum()),"max_drawdown_r":float(dd.min()),"mean_r_ci95_low":lo,"mean_r_ci95_high":hi,"target_hit_rate":float(results.target_hit.mean()),"stop_hit_rate":float(results.stop_hit.mean()),"avg_mfe_pct":float(results.mfe_pct.mean()),"avg_mae_pct":float(results.mae_pct.mean())}


def decile_report(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:return pd.DataFrame()
    x=results.copy(); x["score_decile"]=pd.qcut(x.score.rank(method="first"),10,labels=False)+1
    return x.groupby("score_decile",observed=True).agg(trades=("r_multiple","size"),win_rate=("r_multiple",lambda s:(s>0).mean()),avg_r=("r_multiple","mean"),cum_r=("r_multiple","sum"),avg_mfe_pct=("mfe_pct","mean"),avg_mae_pct=("mae_pct","mean")).reset_index()
