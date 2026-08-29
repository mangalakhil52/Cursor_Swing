"""Point-in-time alpha research utilities for the weekly swing engine.

All features are computed from data available at the signal timestamp. Labels are
forward-looking and must only be used by research/training code, never by the live
scanner. The module intentionally avoids fitting a model here; it creates a stable,
leakage-resistant research dataset that can later feed calibrated tree/linear models.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class AlphaLabel:
    forward_return_5: float
    forward_return_10: float
    mfe_10: float
    mae_10: float
    target_before_stop: int
    triple_barrier: int


def _ret(c: pd.Series, n: int) -> float:
    if len(c) <= n or not np.isfinite(c.iloc[-n-1]) or c.iloc[-n-1] == 0:
        return 0.0
    return float(c.iloc[-1] / c.iloc[-n-1] - 1.0)


def _efficiency(c: pd.Series, n: int = 20) -> float:
    if len(c) <= n:
        return 0.0
    path = float(c.diff().abs().tail(n).sum())
    return float(abs(c.iloc[-1] - c.iloc[-n-1]) / max(path, 1e-12))


def _residual(stock: pd.Series, benchmark: pd.Series | None, n: int = 60) -> float:
    if benchmark is None or len(stock) < n or len(benchmark) < n:
        return 0.0
    s = stock.pct_change().tail(n)
    b = benchmark.pct_change().tail(n)
    j = pd.concat([s.rename("s"), b.rename("b")], axis=1).dropna()
    if len(j) < 30:
        return 0.0
    var = float(j.b.var(ddof=1))
    beta = float(j.s.cov(j.b) / max(var, 1e-12))
    return float((j.s - beta * j.b).tail(20).sum())


def build_features(daily: pd.DataFrame, benchmark: pd.Series | None = None) -> dict[str, float]:
    """Build signal-time features from the supplied history only."""
    c = daily["close"].astype(float)
    h = daily["high"].astype(float)
    l = daily["low"].astype(float)
    v = daily["volume"].astype(float)
    r = c.pct_change()
    atr = (h - l).rolling(14).mean()
    atr_pct = float(atr.iloc[-1] / c.iloc[-1]) if c.iloc[-1] else 0.0
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    vol20 = v.rolling(20).mean()
    return {
        "ret_3": _ret(c, 3), "ret_5": _ret(c, 5), "ret_10": _ret(c, 10),
        "ret_20": _ret(c, 20), "ret_60": _ret(c, 60),
        "vol_20": float(r.rolling(20).std().iloc[-1]) if len(r) >= 21 else 0.0,
        "atr_pct": atr_pct,
        "efficiency_20": _efficiency(c),
        "ema9_gap": float(c.iloc[-1] / ema9.iloc[-1] - 1.0),
        "ema21_gap": float(c.iloc[-1] / ema21.iloc[-1] - 1.0),
        "ema_spread": float(ema9.iloc[-1] / ema21.iloc[-1] - 1.0),
        "volume_ratio": float(v.iloc[-1] / max(vol20.iloc[-1], 1e-12)),
        "range_position_20": float((c.iloc[-1] - l.iloc[:-1].tail(20).min()) / max(h.iloc[:-1].tail(20).max() - l.iloc[:-1].tail(20).min(), 1e-12)),
        "residual_20": _residual(c, benchmark),
        "path_up_fraction_20": float((r.tail(20) > 0).mean()),
        "path_down_fraction_20": float((r.tail(20) < 0).mean()),
    }


def build_label(daily: pd.DataFrame, index: int, direction: str = "LONG", target_pct: float = 0.08, stop_pct: float = 0.04) -> AlphaLabel | None:
    """Create a forward label at `index`; return None if the future window is incomplete."""
    if index < 60 or index + 10 >= len(daily):
        return None
    c = daily["close"].astype(float).to_numpy()
    h = daily["high"].astype(float).to_numpy()
    l = daily["low"].astype(float).to_numpy()
    entry = c[index]
    sign = 1.0 if direction == "LONG" else -1.0
    f5 = sign * (c[index + 5] / entry - 1.0)
    f10 = sign * (c[index + 10] / entry - 1.0)
    future_h = h[index + 1:index + 11]
    future_l = l[index + 1:index + 11]
    if direction == "LONG":
        mfe = float(future_h.max() / entry - 1.0)
        mae = float(future_l.min() / entry - 1.0)
        target = entry * (1 + target_pct)
        stop = entry * (1 - stop_pct)
        hit_t = np.where(future_h >= target)[0]
        hit_s = np.where(future_l <= stop)[0]
    else:
        mfe = float(1.0 - future_l.min() / entry)
        mae = float(1.0 - future_h.max() / entry)
        target = entry * (1 - target_pct)
        stop = entry * (1 + stop_pct)
        hit_t = np.where(future_l <= target)[0]
        hit_s = np.where(future_h >= stop)[0]
    target_first = int(len(hit_t) > 0 and (len(hit_s) == 0 or hit_t[0] < hit_s[0]))
    barrier = 1 if target_first else -1 if len(hit_s) else 0
    return AlphaLabel(f5, f10, mfe, mae, target_first, barrier)


def make_research_frame(daily: pd.DataFrame, benchmark: pd.Series | None = None, direction: str = "LONG", target_pct: float = .08, stop_pct: float = .04) -> pd.DataFrame:
    """Create a walk-forward research frame with point-in-time features and labels."""
    rows = []
    for i in range(60, len(daily) - 10):
        hist = daily.iloc[:i + 1]
        feat = build_features(hist, benchmark.iloc[:i + 1] if benchmark is not None else None)
        label = build_label(daily, i, direction, target_pct, stop_pct)
        if label is None:
            continue
        rows.append({"date": daily.index[i], **feat, **label.__dict__})
    return pd.DataFrame(rows)
