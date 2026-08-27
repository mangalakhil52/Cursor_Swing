# Strategy stress testing

The research stack now includes parameter-perturbation diagnostics and bootstrap confidence intervals.

A strategy should not be promoted because one exact parameter tuple produces the best historical result. The stress layer evaluates nearby parameter values and summarizes the share of scenarios retaining positive expectancy and PF >= 1.

Bootstrap intervals quantify uncertainty around average R. They do not remove non-stationarity, selection bias, or dependence between trades.

Promotion remains based on untouched out-of-sample data; stress diagnostics are evidence about robustness, not a license to tune against the test set.
