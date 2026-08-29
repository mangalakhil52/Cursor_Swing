# Dynamic position sizing

Sizing combines four constraints:

1. Base risk budget per trade.
2. Fractional Kelly from an out-of-sample win probability and payoff ratio.
3. Volatility targeting that reduces risk when realized volatility is high.
4. Drawdown defense: half risk at 8% drawdown and no new risk at 12%.

Position size is finally bounded by stop distance and maximum portfolio exposure.

Kelly is deliberately fractional because probability and payoff estimates are uncertain. It is a sizing aid, not a guarantee of optimal growth.
