# Alpha validation protocol

The alpha research branch uses a strict chronological protocol:

1. Build features only from observations available at signal time.
2. Build forward labels only in research code.
3. Train on earlier observations and test on later observations.
4. Embargo observations around the train/test boundary to reduce leakage from overlapping forward horizons.
5. Evaluate probability calibration with Brier score and calibration bins.
6. Compare the candidate against the incumbent on untouched OOS periods.

Accuracy alone is not a promotion criterion. Promotion requires economic improvement after portfolio constraints and execution costs.

No probability is interpreted as certainty. A 70% estimated probability remains a statistical estimate and can fail materially in a non-stationary market.
