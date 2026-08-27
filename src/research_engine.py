"""Leakage-aware walk-forward research engine for the swing model."""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
import pandas as pd
from src.event_engine import resolve_trade
from src.regime_engine import classify_market

@dataclass(frozen=True)
class ResearchConfig:
    horizon:int=10; stop_pct:float=.05; target_r:float=1.7; slippage_bps:float=10.; min_history:int=120; shorts_require_fno:bool=True

def _atr(df,n=14):
    tr=pd.concat([(df.high-df.low).abs(),(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1); return tr.rolling(n).mean()

def _signal_features(x,benchmark):
    c=x.close.astype(float); r=c.pct_change(); last=float(c.iloc[-1]); ret={f"mom{n}":c.pct_change(n).iloc[-1] for n in (3,5,10,20,60)}
    path=float(c.diff().abs().tail(20).sum()); net=abs(last-float(c.iloc[-21])); er=net/max(path,1e-12); vol=float(r.rolling(20).std().iloc[-1]*math.sqrt(252)); ema21=float(c.ewm(span=21,adjust=False).mean().iloc[-1]); ext=last/ema21-1; vr=float(x.volume.iloc[-1]/max(float(x.volume.rolling(20).mean().iloc[-1]),1.)); hi20=float(x.high.iloc[:-1].tail(20).max()); lo20=float(x.low.iloc[:-1].tail(20).min()); loc=(last-lo20)/max(hi20-lo20,1e-9); residual=0.
    if benchmark is not None and len(benchmark)>=60:
        s=r.tail(60); b=benchmark.pct_change().tail(60); j=pd.concat([s.rename('s'),b.rename('b')],axis=1).dropna()
        if len(j)>=30:
            beta=np.cov(j.s,j.b,ddof=1)[0,1]/max(float(np.var(j.b,ddof=1)),1e-12); residual=float((j.s-beta*j.b).tail(20).sum())
    score=50+25*np.tanh((c.pct_change(20).iloc[-1])/.10)+15*er+10*np.tanh(residual/.03)
    return {**ret,'efficiency':er,'ann_vol':vol,'extension':ext,'volume_ratio':vr,'location':loc,'residual':residual,'score':float(np.clip(score,0,100))}

def _bootstrap_mean(values,n_boot=2000,seed=42):
    if len(values)<2:return(float('nan'),float('nan'))
    rng=np.random.default_rng(seed); means=np.empty(n_boot)
    for i in range(n_boot):means[i]=rng.choice(values,size=len(values),replace=True).mean()
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def load_fno_symbols(path='data/fno_universe.csv')->set[str]:
    p=Path(path)
    if not p.exists():return set()
    try:return set(pd.read_csv(p)['Symbol'].astype(str).str.strip().str.upper())
    except Exception:return set()

def evaluate_symbol(symbol,daily,benchmark=None,cfg=ResearchConfig(),fno_symbols=None):
    d=daily.sort_index().copy(); required={'open','high','low','close','volume'}
    if not required.issubset(d.columns):raise ValueError(f'missing columns: {required-set(d.columns)}')
    fno_symbols=fno_symbols or set(); rows=[]
    for i in range(max(cfg.min_history,60),len(d)-cfg.horizon-1):
        hist=d.iloc[:i+1]; b=benchmark.iloc[:i+1] if benchmark is not None else None; f=_signal_features(hist,b); direction='LONG' if f['mom20']>=0 else 'SHORT'
        if direction=='SHORT' and cfg.shorts_require_fno and symbol.upper() not in fno_symbols:continue
        regime=classify_market(b) if b is not None and len(b)>=60 else 'INSUFFICIENT_HISTORY'
        entry=float(d.open.iloc[i+1]); slip=cfg.slippage_bps/10000; entry*=1+slip if direction=='LONG' else 1-slip; risk=entry*cfg.stop_pct; stop=entry-risk if direction=='LONG' else entry+risk; target=entry+risk*cfg.target_r if direction=='LONG' else entry-risk*cfg.target_r; future=d.iloc[i+1:i+1+cfg.horizon]
        event=resolve_trade(future,entry,direction,stop,target)
        mfe=((future.high.max()/entry-1) if direction=='LONG' else (entry/future.low.min()-1)) if len(future) else 0.; mae=((future.low.min()/entry-1) if direction=='LONG' else (entry/future.high.max()-1)) if len(future) else 0.
        rows.append({'symbol':symbol,'signal_date':d.index[i],'execution_date':d.index[i+1],'direction':direction,'regime':regime,'score':f['score'],'efficiency':f['efficiency'],'ann_vol':f['ann_vol'],'volume_ratio':f['volume_ratio'],'residual':f['residual'],'mfe_pct':mfe*100,'mae_pct':mae*100,'target_hit':event['event']=='TARGET','stop_hit':event['event'].startswith('STOP'),'event':event['event'],'exit_date':event['exit_date'],'r_multiple':event['r_multiple'],'fwd5_pct':(float(d.close.iloc[min(i+5,len(d)-1)])/entry-1)*100*(1 if direction=='LONG' else -1),'fwd10_pct':(float(d.close.iloc[min(i+10,len(d)-1)])/entry-1)*100*(1 if direction=='LONG' else -1)})
    return pd.DataFrame(rows)

def summarize(results):
    if results.empty:return {'trades':0}
    r=results.r_multiple.astype(float); wins=r[r>0]; losses=r[r<0]; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else float('inf'); lo,hi=_bootstrap_mean(r.to_numpy()); eq=r.cumsum(); dd=eq-eq.cummax()
    return {'trades':int(len(r)),'win_rate':float((r>0).mean()),'avg_r':float(r.mean()),'median_r':float(r.median()),'profit_factor':pf,'cum_r':float(r.sum()),'max_drawdown_r':float(dd.min()),'mean_r_ci95_low':lo,'mean_r_ci95_high':hi,'target_hit_rate':float(results.target_hit.mean()),'stop_hit_rate':float(results.stop_hit.mean()),'avg_mfe_pct':float(results.mfe_pct.mean()),'avg_mae_pct':float(results.mae_pct.mean())}

def decile_report(results):
    if results.empty:return pd.DataFrame()
    x=results.copy(); x['score_decile']=pd.qcut(x.score.rank(method='first'),10,labels=False)+1
    return x.groupby('score_decile',observed=True).agg(trades=('r_multiple','size'),win_rate=('r_multiple',lambda s:(s>0).mean()),avg_r=('r_multiple','mean'),cum_r=('r_multiple','sum'),avg_mfe_pct=('mfe_pct','mean'),avg_mae_pct=('mae_pct','mean')).reset_index()