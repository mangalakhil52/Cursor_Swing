# Next Research Stage

The repository now has four research primitives:

- `src/advanced_engine.py`: mathematical feature layer.
- `src/research_engine.py`: point-in-time forward-outcome engine.
- `src/calibration.py`: score/event calibration diagnostics.
- `src/regime_engine.py`: conditional market-regime classification.

## Acceptance criteria before live deployment

A signal family should not be promoted merely because historical cumulative return is positive. Require:

- positive out-of-sample average R;
- profit factor comfortably above 1 after slippage;
- controlled maximum drawdown;
- stable performance across multiple chronological windows;
- no severe degradation in the final untouched test period;
- increasing forward expectancy across score deciles;
- acceptable MFE/MAE profile;
- probability calibration error measured before treating probabilities as probabilities;
- robustness to reasonable changes in thresholds;
- no single symbol or sector responsible for most profits.

## Planned ML stage

Only after the baseline passes should we add a supervised model. Candidate features should be point-in-time only. The preferred workflow is purged/embargoed time-series validation, class-balanced or cost-sensitive learning where justified, probability calibration, feature ablation, and walk-forward retraining. Hyperparameters must never be selected on the final test period.
