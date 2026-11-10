# Validation calibration summary



The lowest Brier score was for SVM (trained on a 50,000-row stratified subsample) (0.10433); the highest was for DecisionTree (0.18409).

The lowest log loss was for SVM (trained on a 50,000-row stratified subsample) (0.33276); log loss is reported as a validation probability score.

Across the nine models, calibration intercepts ranged from -1.73375 to -0.01650, and slopes ranged from 0.68489 to 1.07571.

SVM probabilities exceeded 0.50 for 0.03010 of validation rows, compared with 0.18422 to 0.37247 for the other models.

At the saved SVM threshold of 0.30, its probability exceeded that cutoff for 0.19730 of rows; the other models ranged from 0.17087 to 0.20364 at their own saved thresholds.

Expected calibration error ranged from 0.00878 to 0.23806; ten equal-frequency bins were formed by stable sorting probabilities and splitting rows into ten groups whose counts differ by at most one.

These are descriptive validation results and do not establish why any model's probabilities have a particular distribution.
