# Risk and Model Promotion

## Position sizing

The risk engine sizes positions from cash risk divided by entry-to-stop distance and then caps gross exposure. This prevents a tight stop from creating an oversized position.

## Risk-of-ruin

Monte Carlo resampling of historical R outcomes estimates the probability of breaching a defined equity-loss threshold. This is a diagnostic, not a guarantee: historical outcomes are not independent or stationary.

## Model promotion

A candidate model is promoted only when it has enough out-of-sample trades, positive expectancy, acceptable profit factor, and demonstrably improves on the incumbent OOS result. Thresholds must be evaluated on data that was not used to tune the candidate.

The promotion gate defaults to 100 trades, PF >= 1.15 and average R >= 0.05, plus improvement over the incumbent. These are engineering guardrails, not claims of profitability.
