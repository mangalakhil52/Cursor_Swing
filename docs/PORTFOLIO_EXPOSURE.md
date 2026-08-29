# Portfolio exposure controls

The portfolio layer now treats correlated positions as shared risk.

Controls include:
- sector concentration cap;
- market beta cap;
- rolling return correlation cluster cap;
- explicit rejection reasons.

These are portfolio constraints, not alpha predictors. A high-scoring stock can still be rejected when it duplicates an existing risk factor.

Defaults are intentionally conservative and belong in configuration once calibrated against the intended capital base.
