# Expected-edge gate

A candidate is not selected merely because its raw score is high. The final gate estimates expected R from the model probability and conditional win/loss magnitudes:

`E[R] = p(win) * avg_win_R - (1-p(win)) * avg_loss_R`

The candidate must satisfy the minimum score, calibrated probability, reward/risk and positive expected-R thresholds.

This does not mean a trade is guaranteed to run. Markets are stochastic, estimates are uncertain, and the probability must come from out-of-sample/calibrated evidence rather than a hand-written certainty.
