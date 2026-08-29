# Nested validation

Parameter selection must not use the final test period. Each outer fold reserves a completely untouched test window. Parameters such as probability thresholds are selected only on an earlier validation window, with an embargo from training data.

This protects the final OOS estimate from threshold overfitting. A parameter that wins repeatedly in validation but fails in subsequent test windows is evidence of instability and must not be promoted.

The current runner is a baseline implementation. Production promotion should additionally require minimum fold count, minimum trades per fold, stability across adjacent thresholds, and cost-aware portfolio simulation.
