# Signal lifecycle

The scanner now has a persistent registry concept so a signal can be treated as a lifecycle object rather than a repeated daily row.

A signal is identified by session, symbol and direction and can be tracked as NEW, ACTIVE, EXITED or REJECTED by downstream execution/reporting layers. A configurable cooldown prevents repeated entries caused by the same unresolved move.

Regime-aware gating is conservative by default: long entries are not automatically accepted in strong bear regimes, and short entries are not automatically accepted in bull regimes. F&O eligibility remains a separate hard requirement for overnight shorts.
