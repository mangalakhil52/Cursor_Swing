# Research validation suite

Before a research report is considered trustworthy, the pipeline should verify that execution occurs after signal generation, exits occur after execution, R outcomes are finite, and directions are valid.

These checks are deliberately mechanical. They do not prove profitability; they prevent obvious temporal leakage and malformed labels from contaminating the statistical analysis.

The research engine uses next-session open execution and resolves the first stop/target event sequentially. Daily OHLC cannot establish intraday ordering when both levels occur in one candle, so the event engine uses the conservative stop-first convention.
