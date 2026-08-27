# Quantitative Research Protocol

The research engine is deliberately separated from live signal generation. A signal is formed at the close of day `t`; the baseline execution is the open of `t+1`, with configurable slippage. No future prices are used to form the signal.

## Required validation

1. Run the same frozen model over multiple years of NSE history.
2. Split history chronologically into development and untouched test periods.
3. Use walk-forward evaluation rather than random train/test splits.
4. Report 1/3/5/10/15-session forward returns, MFE, MAE, target-hit and stop-hit rates.
5. Evaluate R expectancy, profit factor, drawdown and confidence intervals.
6. Inspect score deciles. A valid ranking model should show monotonic improvement in forward outcomes as score rises; if it does not, the score is not predictive enough.
7. Repeat by market regime, direction and volatility regime.
8. Do not tune thresholds on the final test period.

## Interpretation

`edge_probability` in the live advanced engine is a ranking diagnostic, not a calibrated probability. A probability may only be called calibrated after an out-of-sample calibration study demonstrates that, for example, signals in the 70% bucket actually realize close to 70% positive outcomes under the defined event.

The objective is not maximum historical return. The objective is stable positive expectancy after realistic execution assumptions and across unseen regimes.
