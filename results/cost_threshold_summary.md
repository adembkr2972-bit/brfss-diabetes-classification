# Validation cost-threshold sensitivity

Cost ratios are illustrative sensitivity settings for cost(FN) / cost(FP), not clinical cost estimates.

| Ratio | Lowest-cost model | Minimum FP + ratio * FN | Threshold |
|---:|---|---:|---:|
| 1 | XGBoost | 9347 | 0.84 |
| 2 | MLPClassifier | 15942 | 0.71 |
| 3 | XGBoost | 20428 | 0.66 |
| 5 | LightGBM | 26036 | 0.52 |
| 10 | LightGBM | 33173 | 0.32 |

The full rank ordering for each ratio is in `cost_ranking_by_ratio.csv`; ties in minimum cost receive sequential ranks in model-name order.
The `cost_optimal_thresholds.csv` file gives each model's cost-optimal threshold and its difference from that model's validation macro-F1-optimal threshold.
All counts and costs use the validation probability cache written by `validation_calibration.py` and the threshold grid from 0.05 through 0.95 in 0.01 steps.
These descriptive results do not establish clinical utility or clinical costs.
