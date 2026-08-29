# Adaptive sizing v2

Position sizing now responds to four independent dimensions:

- estimated edge via fractional Kelly;
- realized annualized volatility relative to a target-volatility budget;
- portfolio drawdown state;
- correlation concentration penalty.

The resulting risk budget is bounded by global minimum/maximum risk and maximum notional exposure.

Drawdown controls intentionally reduce risk progressively and stop new risk at 12% drawdown. The Kelly component is fractional because estimated probabilities are uncertain.
