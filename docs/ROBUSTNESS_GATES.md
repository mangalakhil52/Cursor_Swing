# Robustness gates

Production promotion must be based on out-of-sample evidence, not the best backtest configuration.

The robustness gate requires enough independent time blocks and stable positive expectancy across them, plus survival under nearby parameter perturbations. Failure means the candidate remains research-only.

A live drawdown guard is also available: normal risk below 8% drawdown, half risk from 8% to <12%, and halt new entries at 12% or worse. Existing positions remain subject to their predefined exits; this guard does not move stops after entry.
