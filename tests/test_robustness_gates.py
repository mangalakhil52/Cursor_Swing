from src.robustness_gates import promotion_evidence

def test_failed_evidence_blocks_promotion():
    x=promotion_evidence(folds=2,positive_fold_share=.5,median_to_best=.4,positive_config_share=.4,oos_expectancy=.05)
    assert x['promote'] is False and x['failed_checks']

def test_strong_evidence_passes():
    x=promotion_evidence(folds=6,positive_fold_share=.8,median_to_best=.7,positive_config_share=.8,oos_expectancy=.2)
    assert x['promote'] is True
