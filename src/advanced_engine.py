"""Advanced quantitative swing engine.

Designed as a research layer, not a claim of certainty.  It combines
volatility-normalised momentum, trend efficiency, persistence, entropy,
mean-reversion pressure, volume surprise, drawdown context, and tail-risk
checks.  All calculations use information available at the signal close.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdvancedSignal:
    score: float
    edge_probability: float
    regime_score: float
    momentum_score: float
    trend_score: float
    flow_score: float
    structure_score: float
    risk_score: float
    expected_r_multiple: float
    volatility_regime: str
    persistence: float
    entropy: float
    efficiency_ratio: float
    residual_strength: float
    reasons: tuple[str, ...]
    reject_reasons: tuple[str, ...]


def _ret(c: pd.Series, n: int) -> float:
    if len(c) <= n or float(c.iloc[-n - 1]) == 0:
        return 0.0
    return float(c.iloc[-1] / c.iloc[-n - 1] - 1.0)


def _z(value: float, series: pd.Series, floor: float = 1e-9) -> float:
    s = series.dropna().astype(float)
    if len(s) < 10:
        return 0.0
    sd = float(s.std(ddof=1))
    return (value - float(s.mean())) / max(sd, floor)


def _entropy(returns: pd.Series) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 20:
        return 1.0
    # Four-state sign/size discretisation. Lower entropy = more ordered path.
    q = r.quantile([.25, .5, .75]).to_numpy()
    states = np.digitize(r.to_numpy(), q, right=True)
    counts = np.bincount(states, minlength=4).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum() / 2.0)


def _hurst_proxy(returns: pd.Series) -> float:
    r = returns.dropna().astype(float).to_numpy()
    if len(r) < 30:
        return 0.5
    # R/S slope across several horizons; intentionally a proxy, not a formal H estimator.
    vals = []
    ns = [8, 12, 16, 24, min(32, len(r))]
    for n in ns:
        if n > len(r):
            continue
        x = r[-n:]
        y = x - x.mean()
        rs = (np.cumsum(y).max() - np.cumsum(y).min()) / max(y.std(ddof=1), 1e-12)
        if rs > 0:
            vals.append((math.log(n), math.log(rs)))
    if len(vals) < 2:
        return 0.5
    return float(np.polyfit(np.array(vals)[:, 0], np.array(vals)[:, 1], 1)[0])


def _efficiency(close: pd.Series, n: int = 20) -> float:
    if len(close) <= n:
        return 0.0
    path = close.diff().abs().tail(n).sum()
    net = abs(float(close.iloc[-1] - close.iloc[-n - 1]))
    return float(net / max(float(path), 1e-12))


def _beta_residual_strength(stock: pd.Series, bench: pd.Series | None, n: int = 60) -> float:
    if bench is None or len(stock) < n or len(bench) < n:
        return 0.0
    s = stock.pct_change().tail(n).dropna()
    b = bench.pct_change().tail(n).dropna()
    joined = pd.concat([s.rename("s"), b.rename("b")], axis=1).dropna()
    if len(joined) < 30:
        return 0.0
    beta = np.cov(joined["s"], joined["b"], ddof=1)[0, 1] / max(np.var(joined["b"], ddof=1), 1e-12)
    alpha = joined["s"] - beta * joined["b"]
    return float(alpha.tail(20).sum() * 100.0)


def compute_advanced(daily: pd.DataFrame, benchmark: pd.Series | None, direction: str) -> AdvancedSignal:
    c = daily["close"].astype(float)
    h = daily["high"].astype(float)
    l = daily["low"].astype(float)
    v = daily["volume"].astype(float)
    r = c.pct_change()
    atr = (h - l).rolling(14).mean()
    atr_pct = float(atr.iloc[-1] / c.iloc[-1] * 100) if c.iloc[-1] else 0.0

    # Multi-horizon momentum with volatility normalisation.
    m = np.array([_ret(c, n) for n in (3, 5, 10, 20, 60)])
    vol20 = float(r.rolling(20).std().iloc[-1] * math.sqrt(252)) if r.rolling(20).std().iloc[-1] == r.rolling(20).std().iloc[-1] else 0.0
    direction_sign = 1.0 if direction == "LONG" else -1.0
    momentum = direction_sign * float(np.dot(m, np.array([.08, .14, .22, .31, .25])))
    momentum_z = float(np.tanh(momentum / max(vol20, .03)) * 50 + 50)

    er = _efficiency(c, 20)
    hurst = _hurst_proxy(r.tail(80))
    ent = _entropy(r.tail(60))
    persistence = max(0.0, min(1.0, 0.55 * er + 0.45 * max(0.0, min(1.0, (hurst - .35) / .45))))
    trend = 100.0 * (0.60 * er + 0.40 * persistence)

    # Volume surprise and signed-flow proxy.
    vmean = v.rolling(20).mean()
    vs = float(v.iloc[-1] / max(vmean.iloc[-1], 1.0))
    signed = float((np.sign(r.tail(20)) * v.tail(20)).sum() / max(v.tail(20).sum(), 1.0))
    flow = 50.0 + 28.0 * math.tanh((vs - 1.0) / .7) + 22.0 * direction_sign * signed
    flow = max(0.0, min(100.0, flow))

    # Structure: breakout distance, pullback depth and close location.
    hi20 = float(h.iloc[:-1].tail(20).max()) if len(h) > 21 else float(h.max())
    lo20 = float(l.iloc[:-1].tail(20).min()) if len(l) > 21 else float(l.min())
    if direction == "LONG":
        location = (float(c.iloc[-1]) - lo20) / max(hi20 - lo20, 1e-9)
        breakout = (float(c.iloc[-1]) - hi20) / max(float(atr.iloc[-1]), 1e-9)
    else:
        location = (hi20 - float(c.iloc[-1])) / max(hi20 - lo20, 1e-9)
        breakout = (lo20 - float(c.iloc[-1])) / max(float(atr.iloc[-1]), 1e-9)
    structure = max(0.0, min(100.0, 45.0 + 35.0 * (location - .5) + 20.0 * math.tanh(breakout)))

    # Residual alpha attempts to remove broad-index beta from the move.
    residual = direction_sign * _beta_residual_strength(c, benchmark)
    residual_score = 50.0 + 50.0 * math.tanh(residual / 3.0)

    # Volatility regime: avoid both dead stocks and unstable tails.
    atr_history = (atr / c * 100).dropna().tail(100)
    atr_z = _z(atr_pct, atr_history)
    if atr_z > 2.0 or atr_pct > 8.0:
        vol_regime = "EXTREME"
        risk_score = 25.0
    elif atr_z < -1.5 or atr_pct < 1.2:
        vol_regime = "COMPRESSED"
        risk_score = 35.0
    elif abs(atr_z) <= 1.25:
        vol_regime = "NORMAL"
        risk_score = 85.0
    else:
        vol_regime = "EXPANDING"
        risk_score = 72.0

    # Tail/overextension penalty.
    dd20 = float(c.iloc[-1] / c.tail(20).max() - 1.0)
    extension = abs(float(c.iloc[-1] / c.ewm(span=21, adjust=False).mean().iloc[-1] - 1.0))
    if extension > .12:
        risk_score -= 30
    if direction == "LONG" and dd20 < -.08:
        risk_score -= 10
    if direction == "SHORT" and dd20 > -.02:
        risk_score -= 5
    risk_score = max(0.0, min(100.0, risk_score))

    regime = 50.0
    if persistence > .65 and ent < .85:
        regime += 20
    if vol_regime in ("NORMAL", "EXPANDING"):
        regime += 15
    if residual_score > 65:
        regime += 15
    regime = max(0.0, min(100.0, regime))

    score = (
        .23 * momentum_z
        + .20 * trend
        + .16 * flow
        + .14 * structure
        + .12 * residual_score
        + .10 * regime
        + .05 * risk_score
    )
    edge = 0.50 + 0.47 * math.tanh((score - 68.0) / 13.0)
    expected_r = max(-0.5, min(3.5, (edge - .5) * 5.0))

    reasons = []
    rejects = []
    if persistence >= .65:
        reasons.append(f"persistent trend path (efficiency {er:.2f}, Hurst proxy {hurst:.2f})")
    if residual_score >= 65:
        reasons.append(f"positive residual alpha after index beta ({residual:.2f}%)")
    if flow >= 70:
        reasons.append(f"volume/flow expansion ({vs:.2f}x volume, signed flow {signed:+.2f})")
    if structure >= 70:
        reasons.append("strong price-location / breakout structure")
    if ent <= .70:
        reasons.append(f"low return-path entropy ({ent:.2f})")
    if vol_regime == "EXTREME":
        rejects.append("extreme volatility regime")
    if extension > .12:
        rejects.append("price extension >12% from 21 EMA")
    if risk_score < 35:
        rejects.append("poor volatility/tail-risk profile")
    if score < 68:
        rejects.append(f"advanced composite below threshold ({score:.1f})")

    return AdvancedSignal(
        score=round(float(score), 2),
        edge_probability=round(float(edge), 4),
        regime_score=round(float(regime), 2),
        momentum_score=round(float(momentum_z), 2),
        trend_score=round(float(trend), 2),
        flow_score=round(float(flow), 2),
        structure_score=round(float(structure), 2),
        risk_score=round(float(risk_score), 2),
        expected_r_multiple=round(float(expected_r), 2),
        volatility_regime=vol_regime,
        persistence=round(float(persistence), 3),
        entropy=round(float(ent), 3),
        efficiency_ratio=round(float(er), 3),
        residual_strength=round(float(residual), 3),
        reasons=tuple(reasons),
        reject_reasons=tuple(rejects),
    )
