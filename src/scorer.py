"""Quantitative weekly swing scorer with an advanced mathematical overlay."""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from src.constants import CONVICTION_C,DIRECTION_LONG,DIRECTION_SHORT,MODE_SWING,SETUP_BASE_BREAK,SETUP_BREAKOUT,SETUP_DISPLACEMENT_CONTINUATION,SETUP_EMA_RECLAIM,SETUP_FAIR_VALUE_REVERSION,SETUP_RS_LEADER,SETUP_TREND_PULLBACK
from src.data_fetcher import MarketSnapshot
from src.indicators import enrich_daily,pivot_levels
from src.intelligence import MarketContext,StockIntelligence,analyze_swing_stock,conviction_grade
from src.advanced_engine import compute_advanced

@dataclass
class AnalysisResult:
    symbol:str; direction:str; setup:str; score:float; conviction:str; confluence:int
    trend_score:float; momentum_score:float; volume_score:float; volatility_score:float; market_score:float; energy_score:float; rs_score:float
    entry:float; trigger:float; support:float; resistance:float; atr_value:float; gap_pct:float; day_change_pct:float; volume_ratio:float; rsi:float; relative_strength:float; session_date:object; intel:StockIntelligence
    reasons:list[str]=field(default_factory=list); risks:list[str]=field(default_factory=list); thesis:str=""; hold_horizon:str="5-10 trading days"
    advanced_score:float=0.0; edge_probability:float=0.0; expected_r_multiple:float=0.0; regime_score:float=0.0; persistence:float=0.0; entropy:float=0.0; efficiency_ratio:float=0.0; residual_strength:float=0.0; volatility_regime:str=""; advanced_reasons:list[str]=field(default_factory=list); advanced_rejects:list[str]=field(default_factory=list)

class TradeScorer:
    def __init__(self,config:dict,market:MarketContext,nifty_closes:pd.Series|None=None,mode:str=MODE_SWING)->None:
        self.config=config; self.market=market; self.mode=mode; self.nifty_bias=market.bias; self.nifty_closes=nifty_closes; self.weights=config["scoring"]["weights"]; ind=config["indicators"]; self.ema_fast=ind["ema_fast"]; self.ema_slow=ind["ema_slow"]; self.rsi_period=ind["rsi_period"]; self.atr_period=ind["atr_period"]; self.advanced_cfg=config.get("advanced",{})

    def analyze(self,snapshot:MarketSnapshot)->AnalysisResult|None:
        daily=enrich_daily(snapshot.daily,self.ema_fast,self.ema_slow,self.rsi_period,self.atr_period)
        if len(daily)<80:return None
        row=daily.iloc[-1]; prev=daily.iloc[-2]; f=self.config["filters"]; price=float(row["close"])
        if not(f["min_price"]<=price<=f["max_price"]):return None
        avg_vol=float(row["vol_sma"]) if pd.notna(row["vol_sma"]) else 0.0
        if avg_vol<f["min_avg_volume"] or price*avg_vol/1e7<f["min_dollar_volume_cr"]:return None
        atr=float(row["atr"]) if pd.notna(row["atr"]) else price*.02; rsi=float(row["rsi"]) if pd.notna(row["rsi"]) else 50.; vr=float(row["volume"]/avg_vol) if avg_vol else 1.
        intel=analyze_swing_stock(daily=daily,atr_value=atr,last_price=snapshot.last_price,volume_ratio=vr,market=self.market,nifty_closes=self.nifty_closes)
        if intel.is_dead:return None
        video=self.config.get("video_methodology",{})
        if video.get("enabled",True) and video.get("reject_grade_b",True) and intel.setup_grade=="B":return None
        if intel.atr_pct<float(self.config.get("intelligence",{}).get("min_atr_pct",1.2)):return None
        min_rs=float(self.config.get("intelligence",{}).get("min_rs_20d",1.5))
        if abs(intel.rs_20d)<min_rs and intel.breakout_quality<70 and intel.pullback_quality<70:return None
        trend,d=self._trend(intel,rsi); mom=self._momentum(rsi,intel); vol=self._volume(vr); vola=self._volatility(intel); market=self._market(d); energy=max(intel.pullback_quality,intel.breakout_quality,intel.trend_quality*.8); rs=self._rs(intel,d)
        setups=self._setups(intel,d,rsi,vr,snapshot)
        if not setups:return None
        setup,direction,setup_reasons=setups[0]
        adv=compute_advanced(daily,self.nifty_closes,direction)
        if self.advanced_cfg.get("enabled",True):
            if self.advanced_cfg.get("hard_reject",True) and adv.reject_reasons:return None
            if adv.score<float(self.advanced_cfg.get("min_score",68)) or adv.edge_probability<float(self.advanced_cfg.get("min_edge_probability",.60)):return None
        w=self.weights; base=(trend*w.get("trend",18)+mom*w.get("momentum",12)+vol*w.get("volume",12)+vola*w.get("volatility",10)+market*w.get("market_alignment",12)+energy*w.get("energy",16)+rs*w.get("relative_strength",20))/100
        composite=min(100.,.45*base+.55*adv.score+(4 if intel.setup_grade=="A+" else 2 if intel.setup_grade=="A" else 0))
        confluence=len(self._independent_factors(intel,d,rsi,vr)); conviction=conviction_grade(composite,intel,max(1,confluence)); min_score=float(self.config.get("scoring",{}).get("min_score",60))
        if composite<min_score or conviction==CONVICTION_C:return None
        piv=pivot_levels(float(prev["high"]),float(prev["low"]),float(prev["close"])); entry=snapshot.last_price; hi20=float(daily["high"].iloc[:-1].tail(20).max()); lo20=float(daily["low"].iloc[:-1].tail(20).min()); hi50=float(daily["high"].iloc[:-1].tail(50).max()); lo50=float(daily["low"].iloc[:-1].tail(50).min()); ph=float(prev["high"]); pl=float(prev["low"])
        if direction==DIRECTION_LONG:
            trigger=max(entry,hi20*1.002) if setup in (SETUP_BREAKOUT,SETUP_BASE_BREAK) else entry; ss=[x for x in (pl,intel.ema_fast,intel.ema_slow,piv["s1"],lo20) if x<trigger]; rr=[x for x in (ph,hi20,hi50,piv["r1"],piv["r2"],intel.fair_value if setup==SETUP_FAIR_VALUE_REVERSION else 0) if x>trigger]; support=max(ss) if ss else 0.; resistance=min(rr) if rr else 0.
        else:
            trigger=min(entry,lo20*.998) if setup in (SETUP_BREAKOUT,SETUP_BASE_BREAK) else entry; rr=[x for x in (ph,intel.ema_fast,intel.ema_slow,piv["r1"],hi20) if x>trigger]; ss=[x for x in (pl,lo20,lo50,piv["s1"],piv["s2"],intel.fair_value if setup==SETUP_FAIR_VALUE_REVERSION else 0) if x<trigger]; resistance=min(rr) if rr else 0.; support=max(ss) if ss else 0.
        hold=str(self.config.get("swing",{}).get("hold_horizon","5-10 trading days")); reasons=setup_reasons+intel.notes+list(adv.reasons); risks=self._risks(intel,adv); thesis=f"{snapshot.symbol}: {direction.lower()} {setup.replace('_',' ').title()}, advanced score {adv.score:.1f}, estimated model edge {adv.edge_probability:.0%}."
        return AnalysisResult(symbol=snapshot.symbol,direction=direction,setup=setup,score=round(composite,1),conviction=conviction,confluence=confluence,trend_score=round(trend,1),momentum_score=round(mom,1),volume_score=round(vol,1),volatility_score=round(vola,1),market_score=round(market,1),energy_score=round(energy,1),rs_score=round(rs,1),entry=entry,trigger=float(trigger),support=float(support),resistance=float(resistance),atr_value=atr,gap_pct=round(snapshot.gap_pct,2),day_change_pct=round(snapshot.day_change_pct,2),volume_ratio=round(vr,2),rsi=round(rsi,1),relative_strength=intel.rs_20d,session_date=snapshot.session_date,intel=intel,reasons=reasons,risks=risks,thesis=thesis,hold_horizon=hold,advanced_score=adv.score,edge_probability=adv.edge_probability,expected_r_multiple=adv.expected_r_multiple,regime_score=adv.regime_score,persistence=adv.persistence,entropy=adv.entropy,efficiency_ratio=adv.efficiency_ratio,residual_strength=adv.residual_strength,volatility_regime=adv.volatility_regime,advanced_reasons=list(adv.reasons),advanced_rejects=list(adv.reject_reasons))

    def _trend(self,i,r):
        if i.trend_quality>=80:return i.trend_quality,DIRECTION_LONG if i.ema_fast>i.ema_slow else DIRECTION_SHORT
        if i.ema_fast>i.ema_slow:return max(55.,i.trend_quality),DIRECTION_LONG
        if i.ema_fast<i.ema_slow:return max(55.,i.trend_quality),DIRECTION_SHORT
        return 45.,DIRECTION_LONG if r>=55 else DIRECTION_SHORT
    def _momentum(self,r,i):return min(100.,(85 if 45<=r<=65 else 70 if 35<=r<45 or 65<r<=72 else 35 if r>78 or r<28 else 55)+(8 if abs(i.rs_5d)>=1.5 else 0))
    def _volume(self,v):return 92. if v>=1.8 else 80. if v>=1.3 else 60. if v>=1 else 35.
    def _volatility(self,i):return 85. if 1.5<=i.atr_pct<=5 else 60. if 1.2<=i.atr_pct<=7 else 30.
    def _market(self,d):return 55. if self.nifty_bias=="NEUTRAL" else 90. if (d==DIRECTION_LONG and self.nifty_bias=="BULLISH") or (d==DIRECTION_SHORT and self.nifty_bias=="BEARISH") else 30.
    def _rs(self,i,d):
        x=i.rs_20d
        if d==DIRECTION_LONG:return 95. if x>=5 else 85. if x>=3 else 70. if x>=1.5 else 50. if x>=0 else 25.
        return 95. if x<=-5 else 85. if x<=-3 else 70. if x<=-1.5 else 50. if x<=0 else 25.
    def _independent_factors(self,i,d,r,v):
        out=[]
        if (d==DIRECTION_LONG and i.ema_fast>i.ema_slow) or (d==DIRECTION_SHORT and i.ema_fast<i.ema_slow):out.append("trend")
        if abs(i.rs_20d)>=3:out.append("RS")
        if i.breakout_quality>=70 or i.pullback_quality>=70:out.append("structure")
        if v>=1.3:out.append("volume")
        if 45<=r<=72:out.append("momentum")
        return out
    def _setups(self,i,d,r,v,s):
        f=[]; th=float(self.config.get("video_methodology",{}).get("min_fair_value_distance_atr",.8))
        if i.setup_grade in ("A+","A"):
            if i.fair_value_distance_atr>=th and i.displacement_direction==DIRECTION_SHORT:f.append((SETUP_FAIR_VALUE_REVERSION,DIRECTION_SHORT,["displacement mean-reversion"]));
            elif i.fair_value_distance_atr<=-th and i.displacement_direction==DIRECTION_LONG:f.append((SETUP_FAIR_VALUE_REVERSION,DIRECTION_LONG,["displacement mean-reversion"]))
            if i.displacement_direction==d and ((d==DIRECTION_LONG and i.fair_value_distance_atr>0) or (d==DIRECTION_SHORT and i.fair_value_distance_atr<0)):f.append((SETUP_DISPLACEMENT_CONTINUATION,d,["displacement continuation"]))
        if d==DIRECTION_LONG and i.rs_20d>=3 and v>=1:f.append((SETUP_RS_LEADER,d,["RS leader"]))
        if d==DIRECTION_SHORT and i.rs_20d<=-3 and v>=1:f.append((SETUP_RS_LEADER,d,["RS laggard"]))
        if i.pullback_quality>=70 and ((d==DIRECTION_LONG and r<65) or (d==DIRECTION_SHORT and r>35)):f.append((SETUP_TREND_PULLBACK,d,["21 EMA trend pullback"]))
        if i.breakout_quality>=70:f.append((SETUP_BREAKOUT,d,["20-day structure breakout/breakdown"]))
        if d==DIRECTION_LONG and -1<=i.dist_from_ema_pct<=2.5 and s.day_change_pct>0 and r>=48:f.append((SETUP_EMA_RECLAIM,d,["21 EMA reclaim"]))
        if i.breakout_quality>=65 and v>=1.4 and abs(i.rs_5d)>=1:f.append((SETUP_BASE_BREAK,d,["base expansion"]))
        return f
    @staticmethod
    def _risks(i,a):
        r=[]
        if abs(i.dist_from_ema_pct)>6:r.append("extended from 21 EMA")
        if i.rs_5d*i.rs_20d<0:r.append("RS divergence")
        if a.entropy>.9:r.append("high path entropy")
        if a.persistence<.4:r.append("weak persistence")
        return r or ["honor structural stop; no averaging"]

def detect_nifty_bias(df:pd.DataFrame)->str:
    if df.empty or len(df)<21:return "NEUTRAL"
    c=df["close"].astype(float); last=float(c.iloc[-1]); e9=float(c.ewm(span=9,adjust=False).mean().iloc[-1]); e21=float(c.ewm(span=21,adjust=False).mean().iloc[-1]); week=(last/float(c.iloc[-6])-1)*100
    if last>e9>e21 and week>-1:return "BULLISH"
    if last<e9<e21 and week<1:return "BEARISH"
    return "NEUTRAL"
