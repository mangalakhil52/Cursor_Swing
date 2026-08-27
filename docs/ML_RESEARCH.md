# ML Layer

The ML layer is deliberately downstream of the deterministic quantitative engine.

## Features

Current point-in-time features include multi-horizon momentum, efficiency, annualized volatility, volume surprise, beta-adjusted residual strength and the baseline composite score.

## Validation

`PurgedWalkForward` trains chronologically and leaves an embargo between train and test observations. This reduces leakage from overlapping forward-return labels.

## Important limitation

`ml_probability` is an uncalibrated model output until calibration is run on an untouched validation period. It must not be presented to users as a literal probability of profit. The final test period remains frozen while feature engineering and threshold selection are performed.

## Promotion rule

Do not replace the deterministic model with ML unless the ML layer improves out-of-sample expectancy after slippage and remains stable across chronological windows and market regimes.
