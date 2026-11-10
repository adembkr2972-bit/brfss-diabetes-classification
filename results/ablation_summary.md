# Validation weighted-versus-unweighted ablation

The table reports weighted minus unweighted differences for each model. Confidence intervals are paired respondent bootstrap percentile intervals.

| Model | Best-threshold macro-F1 difference | 95% percentile CI |
|---|---:|---:|
| LogisticRegression | 0.00073 | [-0.00080, 0.00214] |
| DecisionTree | -0.00402 | [-0.00999, 0.00158] |
| RandomForest | -0.00115 | [-0.00420, 0.00154] |
| ExtraTrees | 0.00149 | [-0.00134, 0.00419] |
| SVM (trained on a 50,000-row stratified subsample) | 0.00696 | [0.00311, 0.01076] |
| XGBoost | -0.00040 | [-0.00251, 0.00165] |
| LightGBM | -0.00043 | [-0.00291, 0.00187] |
| CatBoost | -0.00099 | [-0.00327, 0.00135] |
| MLPClassifier | -0.00043 | [-0.00293, 0.00194] |

The 95% interval for best-threshold macro F1 was wholly above zero for: SVM (trained on a 50,000-row stratified subsample). It was wholly below zero for: none. It included zero for: LogisticRegression, DecisionTree, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost, MLPClassifier. These intervals describe paired respondent resampling on this validation sample and are not BRFSS design-based intervals. Threshold selection can compensate for weighting, so a small best-threshold difference does not imply weighting was pointless. For macro-F1 bootstrap replicates, each arm's threshold was held fixed at the value selected on the full validation set; these intervals are conditional on those thresholds and do not include threshold-selection uncertainty.

Bootstrap resamples: 1000; seed: 42.
