# Execution-path simulation

Daily OHLC data cannot reveal the exact intraday order of a stop and target. The simulator therefore resolves ambiguous same-bar stop/target touches conservatively as stop-first.

Overnight gaps are handled at the opening price when the open has crossed the stop/target, rather than assuming an ideal fill at the requested level.

Target-1 books half the position and moves the remaining stop to entry. Target-2 resolves the remaining half. If neither barrier resolves, the position is marked TIME_EXIT at the last available close.

These assumptions are deliberately conservative and should be replaced with intraday data when available.
