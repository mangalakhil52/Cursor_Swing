# Ensemble shadow mode

The ensemble combines the existing structural score with calibrated out-of-sample alpha probability. Regime-specific weights allow the relative contribution of structural persistence and learned alpha to vary with market state.

The ensemble is research/shadow-only until it demonstrates incremental OOS improvement versus the incumbent after costs and portfolio constraints. No production signal is replaced merely because the ensemble score is higher.

For shorts, this score is downstream of the hard F&O eligibility gate; an ensemble score can never override the F&O-only requirement.
