#!/usr/bin/env python3
"""Build point-in-time features shared by ranking and downstream OOS models.

All features use information available on or before the observation date.  The
benchmark is used only for relative-strength and regime context.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def _naive_dates(s):
    x = pd.to_datetime(s)
    return x.dt.tz_localize(None) if getattr(x.dt, "tz", None) is not None else x


def _clip01(x):
    return pd.to_numeric(x, errors="coerce").clip(0.0, 1.0)


def _regime(bench_close: pd.Series, dt: pd.Timestamp) -> str:
    x = bench_close.loc[bench_close.index <= dt]
    if len(x) < 61:
        return "SIDEWAYS"
    ret = x.pct_change()
    r20 = x.iloc[-1] / x.iloc[-21] - 1.0
    r60 = x.iloc[-1] / x.iloc[-61] - 1.0
    v20 = ret.tail(20).std()
    v120 = ret.tail(120).std()
    if v20 > 1.5 * max(v120, 1e-9):
        return "HIGH_VOL_SIDEWAYS"
    if r20 > 0.03 and r60 > 0.06:
        return "STRONG_BULL"
    if r20 > 0.01 and r60 > 0.02:
        return "BULL"
    if r20 < -0.03 and r60 < -0.06:
        return "STRONG_BEAR"
    if r20 < -0.01 and r60 < -0.02:
        return "BEAR"
    return "SIDEWAYS"


def build_features(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = _naive_dates(d["date"])
    b = benchmark.copy()
    b.index = pd.to_datetime(b.index)
    if getattr(b.index, "tz", None) is not None:
        b.index = b.index.tz_localize(None)
    b = b.sort_index()
    close = pd.to_numeric(b["close"], errors="coerce")
    bret20 = close / close.shift(20) - 1.0
    bret60 = close / close.shift(60) - 1.0

    bm = pd.DataFrame({
        "benchmark_ret_20": bret20,
        "benchmark_ret_60": bret60,
        "market_regime": [_regime(close, dt) for dt in close.index],
    })
    bm.index.name = "date"
    d = d.merge(bm.reset_index(), on="date", how="left")

    # Structural quality: trend alignment + path efficiency + position in range.
    ema_alignment = 0.5 + 4.0 * pd.to_numeric(d.get("ema_spread", 0.0), errors="coerce")
    efficiency = _clip01(d.get("efficiency_20", 0.0))
    range_pos = _clip01(d.get("range_position_20", 0.5))
    d["structural_score"] = 0.40 * efficiency + 0.35 * _clip01(ema_alignment) + 0.25 * range_pos

    # Momentum acceleration: recent return versus the 20-bar average pace.
    r5 = pd.to_numeric(d.get("ret_5", 0.0), errors="coerce")
    r20 = pd.to_numeric(d.get("ret_20", 0.0), errors="coerce")
    d["residual_momentum"] = r5 - r20 / 4.0
    d["relative_strength"] = r20 - pd.to_numeric(d["benchmark_ret_20"], errors="coerce")

    # Reward efficient movement relative to realized volatility.
    atr = pd.to_numeric(d.get("atr_pct", 0.03), errors="coerce")
    d["volatility_efficiency"] = efficiency / atr.clip(lower=0.005)

    # Regime fit is deliberately long-only: positive momentum is preferred in
    # bull/bear regimes, while sideways markets favour controlled momentum.
    regime = d["market_regime"].fillna("SIDEWAYS").astype(str)
    momentum_fit = _clip01(0.5 + 3.0 * r20)
    sideways_fit = _clip01(1.0 - r20.abs() / 0.10)
    d["regime_fit"] = np.where(regime.isin(["BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR"]), momentum_fit, sideways_fit)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/alpha_oos_enriched.csv")
    ap.add_argument("--benchmark", default="data/history/_benchmark.csv")
    ap.add_argument("--output", default="reports/alpha_oos_features.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.input)
    b = pd.read_csv(a.benchmark, index_col=0, parse_dates=True)
    out = build_features(d, b)
    required = ["structural_score", "residual_momentum", "relative_strength", "volatility_efficiency", "regime_fit"]
    missing = [c for c in required if c not in out]
    if missing:
        raise SystemExit(f"Missing engineered features: {missing}")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"rows={len(out)} feature_rows={int(out[required].notna().all(axis=1).sum())}")


if __name__ == "__main__":
    main()
