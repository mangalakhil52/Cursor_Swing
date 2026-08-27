# Portfolio realism

The research stack now separates signal edge from executable portfolio edge.

- Repeated signals on the same symbol/direction can be clustered.
- Concurrent positions are capped.
- Same-symbol overlap is rejected.
- Position sizing is risk-based and exposure-capped.
- Round-trip trading costs are modeled conservatively.
- Short-side cost assumptions can differ for F&O.

The simulator should never claim exact realised costs from daily OHLC alone. Where brokerage/tax details depend on broker, product or turnover, the report should expose assumptions rather than hide them.
