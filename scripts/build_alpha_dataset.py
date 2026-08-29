#!/usr/bin/env python3
"""Build a point-in-time alpha research dataset from cached OHLCV history."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.alpha_model import make_research_frame


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df.sort_index()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="data/history_daily")
    p.add_argument("--output", default="reports/alpha_dataset.parquet")
    p.add_argument("--benchmark", default="^NSEI.csv")
    p.add_argument("--direction", choices=["LONG", "SHORT"], default="LONG")
    p.add_argument("--target-pct", type=float, default=.08)
    p.add_argument("--stop-pct", type=float, default=.04)
    args = p.parse_args()
    root = Path(args.history)
    bench_path = root / args.benchmark
    benchmark = load_csv(bench_path)["close"] if bench_path.exists() else None
    frames = []
    for path in sorted(root.glob("*.csv")):
        if path.name == args.benchmark:
            continue
        try:
            daily = load_csv(path)
            if len(daily) < 80:
                continue
            frame = make_research_frame(daily, benchmark, args.direction, args.target_pct, args.stop_pct)
            if not frame.empty:
                frame.insert(0, "symbol", path.stem)
                frames.append(frame)
        except Exception as exc:
            print(f"skip {path.name}: {exc}")
    if not frames:
        raise SystemExit("No eligible history files found")
    out = pd.concat(frames, ignore_index=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out):,} observations across {out['symbol'].nunique():,} symbols to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
