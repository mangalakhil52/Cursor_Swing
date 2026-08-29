"""Strict evidence gates before promoting a strategy to production."""
from __future__ import annotations

def promotion_evidence(*, folds:int, min_folds:int=5, positive_fold_share:float, min_positive_fold_share:float=.60, median_to_best:float, min_median_to_best:float=.50, positive_config_share:float, min_positive_config_share:float=.55, oos_expectancy:float, min_oos_expectancy:float=.10)->dict:
    checks={
      'fold_count':folds>=min_folds,
      'positive_fold_share':positive_fold_share>=min_positive_fold_share,
      'parameter_plateau':median_to_best>=min_median_to_best,
      'positive_config_share':positive_config_share>=min_positive_config_share,
      'oos_expectancy':oos_expectancy>=min_oos_expectancy,
    }
    return {'promote':all(checks.values()),'checks':checks,'failed_checks':[k for k,v in checks.items() if not v]}
