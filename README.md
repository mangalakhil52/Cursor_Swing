# Indian Advanced Quant Swing Trade Finder

Finds up to **two high-quality NSE swing candidates** for a 5–10 trading-day horizon. This is a probabilistic research system, not a guarantee that a stock will rise.

## Pipeline

1. Scan the full NSE EQ universe with liquidity filters.
2. Build daily market structure and Nifty regime.
3. Apply displacement/fair-value methodology with B-grade rejection.
4. Apply classical trend, momentum, volume, volatility, market-alignment and relative-strength factors.
5. Apply an advanced mathematical overlay using:
   - multi-horizon volatility-normalised momentum
   - Kaufman-style directional efficiency ratio
   - Hurst-style persistence proxy
   - return-path entropy
   - volume surprise and signed-flow proxy
   - price-location / breakout geometry
   - beta-adjusted residual strength versus Nifty
   - volatility-regime classification
   - extension, drawdown and tail-risk penalties
6. Reject candidates failing advanced hard gates.
7. Build structure/ATR-based risk plans with minimum 1.7R genuine reward/risk.
8. Send only the surviving shortlist to the LLM for final ranking.
9. Track outcomes in Excel.

## Advanced layer

The advanced score is deliberately **not** a magic prediction probability. `edge_probability` is a model score mapped into a bounded probability-like diagnostic and must be calibrated with out-of-sample data before it is treated as a probability.

The system prefers stocks where several properties agree: persistent directional movement, useful volatility, abnormal participation, strong structure, and strength that remains after removing broad Nifty beta.

## Risk / execution

- Risk per trade: 1% of configured capital.
- Maximum position: 25% of capital.
- Structural stop plus ATR buffer.
- Stop width capped at 5%.
- Genuine minimum R:R raised to 1.7.
- No target stretching to manufacture R:R.
- B-grade displacement setups are rejected.

## Important research rule

Do **not** judge this system by win rate alone. Primary evaluation metrics are:

- expectancy in R
- profit factor
- maximum drawdown
- average winner / average loser
- return distribution by advanced-score bucket
- performance by market regime
- walk-forward and out-of-sample performance

At least 100–200 closed trades across multiple market regimes should be collected before treating the revised model as validated.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
python scripts/refresh_universe.py --source nse_equity
```

Set the LLM provider/key in `.env` and `config.yaml` if AI selection is enabled.

## Run

```bash
python main.py
python main.py --capital 200000
```

The scheduled runner can continue to maintain `reports/swing_performance.xlsx`.

## Disclaimer

Educational/research software only. It is not SEBI-registered investment advice, and no algorithm can guarantee that a stock will "surely run". Use paper trading and out-of-sample validation before risking capital.
