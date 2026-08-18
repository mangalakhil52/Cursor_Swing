"""Swing-trade scoring and setup detection (weekly horizon)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.constants import (
    CONVICTION_C,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    MODE_SWING,
    SETUP_BASE_BREAK,
    SETUP_BREAKOUT,
    SETUP_DISPLACEMENT_CONTINUATION,
    SETUP_EMA_RECLAIM,
    SETUP_FAIR_VALUE_REVERSION,
    SETUP_RS_LEADER,
    SETUP_TREND_PULLBACK,
)
from src.data_fetcher import MarketSnapshot
from src.indicators import enrich_daily, pivot_levels
from src.intelligence import (
    MarketContext,
    StockIntelligence,
    analyze_swing_stock,
    conviction_grade,
)


@dataclass
class AnalysisResult:
    symbol: str
    direction: str
    setup: str
    score: float
    conviction: str
    confluence: int
    trend_score: float
    momentum_score: float
    volume_score: float
    volatility_score: float
    market_score: float
    energy_score: float
    rs_score: float
    entry: float
    trigger: float
    support: float
    resistance: float
    atr_value: float
    gap_pct: float
    day_change_pct: float
    volume_ratio: float
    rsi: float
    relative_strength: float
    session_date: object
    intel: StockIntelligence
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    thesis: str = ""
    hold_horizon: str = "5-10 trading days"


class TradeScorer:
    """Scores swing candidates on weekly structure — not same-day noise."""

    def __init__(
        self,
        config: dict,
        market: MarketContext,
        nifty_closes: pd.Series | None = None,
        mode: str = MODE_SWING,
    ) -> None:
        self.config = config
        self.market = market
        self.mode = mode
        self.nifty_bias = market.bias
        self.nifty_closes = nifty_closes
        self.weights = config["scoring"]["weights"]
        ind = config["indicators"]
        self.ema_fast = ind["ema_fast"]
        self.ema_slow = ind["ema_slow"]
        self.rsi_period = ind["rsi_period"]
        self.atr_period = ind["atr_period"]

    def analyze(self, snapshot: MarketSnapshot) -> AnalysisResult | None:
        daily = enrich_daily(
            snapshot.daily,
            self.ema_fast,
            self.ema_slow,
            self.rsi_period,
            self.atr_period,
        )
        if len(daily) < 40:
            return None

        row = daily.iloc[-1]
        prev = daily.iloc[-2]
        filters = self.config["filters"]
        price = float(row["close"])
        if price < filters["min_price"] or price > filters["max_price"]:
            return None

        avg_vol = float(row["vol_sma"]) if pd.notna(row["vol_sma"]) else 0
        if avg_vol < filters["min_avg_volume"]:
            return None
        turnover_cr = (price * avg_vol) / 1e7
        if turnover_cr < filters["min_dollar_volume_cr"]:
            return None

        pivots = pivot_levels(float(prev["high"]), float(prev["low"]), float(prev["close"]))
        atr_value = float(row["atr"]) if pd.notna(row["atr"]) else price * 0.02
        rsi_val = float(row["rsi"]) if pd.notna(row["rsi"]) else 50.0
        vol_ratio = float(row["volume"] / avg_vol) if avg_vol > 0 else 1.0

        intel = analyze_swing_stock(
            daily=daily,
            atr_value=atr_value,
            last_price=snapshot.last_price,
            volume_ratio=vol_ratio,
            market=self.market,
            nifty_closes=self.nifty_closes,
        )
        if intel.is_dead:
            return None

        video_cfg = self.config.get("video_methodology", {})
        if (
            bool(video_cfg.get("enabled", True))
            and bool(video_cfg.get("reject_grade_b", True))
            and intel.setup_grade == "B"
        ):
            return None

        min_atr = float(self.config.get("intelligence", {}).get("min_atr_pct", 1.2))
        min_rs = float(self.config.get("intelligence", {}).get("min_rs_20d", 1.5))
        if intel.atr_pct < min_atr:
            return None
        if abs(intel.rs_20d) < min_rs and intel.breakout_quality < 70 and intel.pullback_quality < 70:
            return None

        trend_score, trend_dir = self._score_trend(intel, rsi_val)
        momentum_score = self._score_momentum(rsi_val, intel)
        volume_score = self._score_volume(vol_ratio)
        volatility_score = self._score_volatility(intel)
        market_score = self._score_market(trend_dir)
        energy_score = max(intel.pullback_quality, intel.breakout_quality, intel.trend_quality * 0.8)
        rs_score = self._score_rs(intel, trend_dir)

        setups = self._detect_setups(intel, trend_dir, rsi_val, vol_ratio, snapshot)
        if not setups:
            return None

        setup, direction, setup_reasons = setups[0]
        confluence = len(setups)

        w = self.weights
        composite = (
            trend_score * w.get("trend", 18)
            + momentum_score * w.get("momentum", 12)
            + volume_score * w.get("volume", 12)
            + volatility_score * w.get("volatility", 10)
            + market_score * w.get("market_alignment", 12)
            + energy_score * w.get("energy", 16)
            + rs_score * w.get("relative_strength", 20)
        ) / 100
        composite = min(100.0, composite + (confluence - 1) * 5)
        if intel.setup_grade == "A+":
            composite = min(100.0, composite + 8)
        elif intel.setup_grade == "A":
            composite = min(100.0, composite + 3)

        conviction = conviction_grade(composite, intel, confluence)
        min_score = float(self.config.get("scoring", {}).get("min_score", 60))
        if conviction == CONVICTION_C and composite < min_score:
            return None

        entry = snapshot.last_price
        prior_high_20 = float(daily["high"].iloc[:-1].tail(20).max())
        prior_low_20 = float(daily["low"].iloc[:-1].tail(20).min())
        prior_high_50 = float(daily["high"].iloc[:-1].tail(50).max())
        prior_low_50 = float(daily["low"].iloc[:-1].tail(50).min())
        previous_high = float(prev["high"])
        previous_low = float(prev["low"])

        if direction == DIRECTION_LONG:
            trigger = max(entry, prior_high_20 * 1.002) if setup in (SETUP_BREAKOUT, SETUP_BASE_BREAK) else entry
            supports = [
                level
                for level in (
                    previous_low,
                    intel.ema_fast,
                    intel.ema_slow,
                    pivots["s1"],
                    prior_low_20,
                )
                if level < trigger
            ]
            resistances = [
                level
                for level in (
                    previous_high,
                    prior_high_20,
                    prior_high_50,
                    pivots["r1"],
                    pivots["r2"],
                    (
                        intel.fair_value
                        if setup == SETUP_FAIR_VALUE_REVERSION
                        else 0.0
                    ),
                )
                if level > trigger
            ]
            # Nearest structure matters; distant 20-day lows produce unusable stops.
            support = max(supports) if supports else 0.0
            resistance = min(resistances) if resistances else 0.0
        else:
            trigger = min(entry, prior_low_20 * 0.998) if setup in (SETUP_BREAKOUT, SETUP_BASE_BREAK) else entry
            resistances = [
                level
                for level in (
                    previous_high,
                    intel.ema_fast,
                    intel.ema_slow,
                    pivots["r1"],
                    prior_high_20,
                )
                if level > trigger
            ]
            supports = [
                level
                for level in (
                    previous_low,
                    prior_low_20,
                    prior_low_50,
                    pivots["s1"],
                    pivots["s2"],
                    (
                        intel.fair_value
                        if setup == SETUP_FAIR_VALUE_REVERSION
                        else 0.0
                    ),
                )
                if level < trigger
            ]
            resistance = min(resistances) if resistances else 0.0
            support = max(supports) if supports else 0.0

        hold = str(self.config.get("swing", {}).get("hold_horizon", "5-10 trading days"))
        reasons = setup_reasons + intel.notes
        risks = self._risks(intel, direction, conviction)
        thesis = (
            f"{snapshot.symbol} is a weekly swing {direction.lower()} via "
            f"{setup.replace('_', ' ').title()}. "
            f"20d RS {intel.rs_20d:+.1f}% vs Nifty, trend quality {intel.trend_quality:.0f}, "
            f"confluence x{confluence}. Hold {hold} — manage on daily closes only."
        )

        return AnalysisResult(
            symbol=snapshot.symbol,
            direction=direction,
            setup=setup,
            score=round(composite, 1),
            conviction=conviction,
            confluence=confluence,
            trend_score=round(trend_score, 1),
            momentum_score=round(momentum_score, 1),
            volume_score=round(volume_score, 1),
            volatility_score=round(volatility_score, 1),
            market_score=round(market_score, 1),
            energy_score=round(energy_score, 1),
            rs_score=round(rs_score, 1),
            entry=entry,
            trigger=float(trigger),
            support=float(support),
            resistance=float(resistance),
            atr_value=atr_value,
            gap_pct=round(snapshot.gap_pct, 2),
            day_change_pct=round(snapshot.day_change_pct, 2),
            volume_ratio=round(vol_ratio, 2),
            rsi=round(rsi_val, 1),
            relative_strength=intel.rs_20d,
            session_date=snapshot.session_date,
            intel=intel,
            reasons=reasons,
            risks=risks,
            thesis=thesis,
            hold_horizon=hold,
        )

    def _score_trend(self, intel: StockIntelligence, rsi: float) -> tuple[float, str]:
        if intel.trend_quality >= 80 and intel.ema_fast > intel.ema_slow:
            return intel.trend_quality, DIRECTION_LONG
        if intel.trend_quality >= 80 and intel.ema_fast < intel.ema_slow:
            return intel.trend_quality, DIRECTION_SHORT
        if intel.ema_fast > intel.ema_slow:
            return max(55.0, intel.trend_quality), DIRECTION_LONG
        if intel.ema_fast < intel.ema_slow:
            return max(55.0, intel.trend_quality), DIRECTION_SHORT
        if rsi >= 55:
            return 45.0, DIRECTION_LONG
        return 45.0, DIRECTION_SHORT

    def _score_momentum(self, rsi: float, intel: StockIntelligence) -> float:
        # Swing sweet spot: not blown-off, not dead
        if 45 <= rsi <= 65:
            base = 85.0
        elif 35 <= rsi < 45 or 65 < rsi <= 72:
            base = 70.0
        elif rsi > 78 or rsi < 28:
            base = 35.0
        else:
            base = 55.0
        if abs(intel.rs_5d) >= 1.5:
            base = min(100.0, base + 8)
        return base

    def _score_volume(self, vol_ratio: float) -> float:
        if vol_ratio >= 1.8:
            return 92.0
        if vol_ratio >= 1.3:
            return 80.0
        if vol_ratio >= 1.0:
            return 60.0
        return 35.0

    def _score_volatility(self, intel: StockIntelligence) -> float:
        # Swing likes 1.5-5% ATR
        if 1.5 <= intel.atr_pct <= 5.0:
            return 85.0
        if 1.2 <= intel.atr_pct < 1.5 or 5.0 < intel.atr_pct <= 7.0:
            return 60.0
        return 30.0

    def _score_market(self, trend_dir: str) -> float:
        if self.nifty_bias == "NEUTRAL":
            return 55.0
        if trend_dir == DIRECTION_LONG and self.nifty_bias == "BULLISH":
            return 90.0
        if trend_dir == DIRECTION_SHORT and self.nifty_bias == "BEARISH":
            return 90.0
        return 30.0

    def _score_rs(self, intel: StockIntelligence, trend_dir: str) -> float:
        rs = intel.rs_20d
        if trend_dir == DIRECTION_LONG:
            if rs >= 5:
                return 95.0
            if rs >= 3:
                return 85.0
            if rs >= 1.5:
                return 70.0
            if rs >= 0:
                return 50.0
            return 25.0
        if rs <= -5:
            return 95.0
        if rs <= -3:
            return 85.0
        if rs <= -1.5:
            return 70.0
        if rs <= 0:
            return 50.0
        return 25.0

    def _detect_setups(
        self,
        intel: StockIntelligence,
        trend_dir: str,
        rsi: float,
        vol_ratio: float,
        snapshot: MarketSnapshot,
    ) -> list[tuple[str, str, list[str]]]:
        found: list[tuple[str, str, list[str]]] = []
        fair_value_threshold = float(
            self.config.get("video_methodology", {}).get(
                "min_fair_value_distance_atr", 0.8
            )
        )

        # Adapted from the video's fair-pricing model to DAILY swing candles.
        # A+ = displacement + break of structure; A = displacement only; B = skip.
        if intel.setup_grade in ("A+", "A"):
            if (
                intel.fair_value_distance_atr >= fair_value_threshold
                and intel.displacement_direction == DIRECTION_SHORT
            ):
                found.append((
                    SETUP_FAIR_VALUE_REVERSION,
                    DIRECTION_SHORT,
                    [
                        f"{intel.setup_grade} displacement back toward fair value "
                        f"INR {intel.fair_value:,.2f} from {intel.fair_value_distance_atr:+.1f} ATR"
                    ],
                ))
            elif (
                intel.fair_value_distance_atr <= -fair_value_threshold
                and intel.displacement_direction == DIRECTION_LONG
            ):
                found.append((
                    SETUP_FAIR_VALUE_REVERSION,
                    DIRECTION_LONG,
                    [
                        f"{intel.setup_grade} displacement back toward fair value "
                        f"INR {intel.fair_value:,.2f} from {intel.fair_value_distance_atr:+.1f} ATR"
                    ],
                ))

            if (
                intel.displacement_direction == trend_dir
                and (
                    (trend_dir == DIRECTION_LONG and intel.fair_value_distance_atr > 0)
                    or (
                        trend_dir == DIRECTION_SHORT
                        and intel.fair_value_distance_atr < 0
                    )
                )
            ):
                found.append((
                    SETUP_DISPLACEMENT_CONTINUATION,
                    trend_dir,
                    [
                        f"{intel.setup_grade} displacement continuation away from fair value; "
                        f"body {intel.displacement_ratio:.1f}x previous"
                    ],
                ))

        if trend_dir == DIRECTION_LONG and intel.rs_20d >= 3 and vol_ratio >= 1.0:
            found.append((
                SETUP_RS_LEADER,
                DIRECTION_LONG,
                [f"20-day RS leader vs Nifty ({intel.rs_20d:+.1f}%) — institutional bid"],
            ))
        if trend_dir == DIRECTION_SHORT and intel.rs_20d <= -3 and vol_ratio >= 1.0:
            found.append((
                SETUP_RS_LEADER,
                DIRECTION_SHORT,
                [f"20-day RS laggard vs Nifty ({intel.rs_20d:+.1f}%)"],
            ))

        if intel.pullback_quality >= 70 and trend_dir == DIRECTION_LONG and rsi < 65:
            found.append((
                SETUP_TREND_PULLBACK,
                DIRECTION_LONG,
                ["Trend pullback into 21 EMA — classic swing long entry"],
            ))
        if intel.pullback_quality >= 70 and trend_dir == DIRECTION_SHORT and rsi > 35:
            found.append((
                SETUP_TREND_PULLBACK,
                DIRECTION_SHORT,
                ["Bear rally into EMA — swing short entry"],
            ))

        if intel.breakout_quality >= 70 and trend_dir == DIRECTION_LONG:
            found.append((
                SETUP_BREAKOUT,
                DIRECTION_LONG,
                ["Swing breakout / hold of 20-day high with volume"],
            ))
        if intel.breakout_quality >= 70 and trend_dir == DIRECTION_SHORT:
            found.append((
                SETUP_BREAKOUT,
                DIRECTION_SHORT,
                ["Breakdown of 20-day low structure"],
            ))

        # EMA reclaim after dip
        if (
            trend_dir == DIRECTION_LONG
            and intel.dist_from_ema_pct >= -1.0
            and intel.dist_from_ema_pct <= 2.5
            and snapshot.day_change_pct > 0
            and rsi >= 48
        ):
            found.append((
                SETUP_EMA_RECLAIM,
                DIRECTION_LONG,
                ["Reclaiming / holding 21 EMA after pullback"],
            ))

        # Tight base then expansion
        if intel.breakout_quality >= 65 and intel.volume_ratio >= 1.4 and abs(intel.rs_5d) >= 1:
            found.append((
                SETUP_BASE_BREAK,
                trend_dir if trend_dir in (DIRECTION_LONG, DIRECTION_SHORT) else DIRECTION_LONG,
                ["Base break with expansion — multi-day continuation candidate"],
            ))

        return found

    @staticmethod
    def _risks(intel: StockIntelligence, direction: str, conviction: str) -> list[str]:
        risks = []
        if conviction == CONVICTION_C:
            risks.append("Lower conviction — half-size only or skip")
        if abs(intel.dist_from_ema_pct) > 6 and direction == DIRECTION_LONG:
            risks.append("Extended from EMA — chasing risk; prefer pullback")
        if intel.rs_5d * intel.rs_20d < 0:
            risks.append("Short-term RS diverging from 20d RS — wait for alignment")
        if not risks:
            risks.append("Honor swing stop on daily close basis — no averaging")
        return risks


def detect_nifty_bias(nifty_df: pd.DataFrame) -> str:
    if nifty_df.empty or len(nifty_df) < 21:
        return "NEUTRAL"
    close = nifty_df["close"].astype(float)
    last = float(close.iloc[-1])
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    week = ((last - float(close.iloc[-6])) / float(close.iloc[-6])) * 100 if len(close) >= 6 else 0
    if last > ema9 > ema21 and week > -1.0:
        return "BULLISH"
    if last < ema9 < ema21 and week < 1.0:
        return "BEARISH"
    return "NEUTRAL"
