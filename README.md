# Indian Advanced Quant Swing Trade Finder

Finds up to **two high-quality NSE swing candidates** for a 5–10 trading-day horizon. This is a probabilistic research system, not a guarantee that a stock will rise.

## Pipeline

1. Scan the NSE EQ universe with liquidity filters.
2. Maintain a current NSE stock-F&O universe separately.
3. Build daily market structure and Nifty regime.
4. Apply displacement/fair-value methodology with B-grade rejection.
5. Apply classical trend, momentum, volume, volatility, market-alignment and relative-strength factors.
6. Apply an advanced mathematical overlay using multi-horizon momentum, directional efficiency, persistence, return-path entropy, flow, breakout geometry, beta-adjusted residual strength, volatility regime and tail-risk penalties.
7. Reject candidates failing advanced hard gates.
8. Build structure/ATR-based risk plans with minimum 1.7R genuine reward/risk.
9. Overnight SHORT signals are permitted **only for stocks currently eligible for NSE stock F&O**. Cash-equity swing shorts are rejected.
10. Send only the surviving shortlist to the LLM for final ranking.
11. Track outcomes and research results.

## F&O short rule

A normal cash-equity delivery swing cannot be held as an overnight naked short. Therefore the scanner treats short setups as a derivatives-only capability. The GitHub Action refreshes `data/fno_universe.csv` from NSE's current permitted-lot-size file before research/scanning. Both live scoring and historical research enforce this gate.

NSE's derivatives framework identifies individual-security futures/options with `FUTSTK` and `OPTSTK` instruments, and eligibility can change over time. The system therefore refreshes the universe rather than maintaining a hard-coded list.

## Advanced layer

The advanced score is deliberately **not** a magic prediction probability. `edge_probability` is a model score mapped into a bounded probability-like diagnostic and must be calibrated with out-of-sample data before it is treated as a probability.

## Research rule

Do not judge this system by win rate alone. Primary metrics are expectancy in R, profit factor, drawdown, MFE/MAE, score-decile monotonicity, market-regime stability, and walk-forward out-of-sample performance.

At least 100–200 closed trades across multiple market regimes should be collected before treating the revised model as validated.

## GitHub-native operation

The repository is designed to run through GitHub Actions. Historical OHLCV data is persisted in `data/history/`: new symbols receive the configured initial lookback, while existing symbols receive only incremental recent data. The research workflow refreshes the NSE equity universe, refreshes F&O eligibility, updates persistent history, runs deterministic research, runs ML walk-forward research, uploads reports as artifacts, and commits changed data/reports back to GitHub.

## Disclaimer

Educational/research software only. It is not SEBI-registered investment advice, and no algorithm can guarantee that a stock will "surely run". Use paper trading and out-of-sample validation before risking capital.
