# Evaluation protocol

1. Freeze the target (`problematic_return`) and the 30-day observation window before modeling.
2. Sort by `order_time` and cut 70% train / 15% validation / 15% test. The test slice is never used for preprocessing, model choice, calibration, or threshold selection.
3. Fit imputers, scalers, and encoders on train only.
4. Compare majority, rule (`historical_return_rate > 0.5`), logistic regression, and histogram gradient boosting.
5. Calibrate the champion on validation (isotonic).
6. Choose the probability threshold that minimizes **validation expected decision cost**.
7. Compute headline precision, recall, PR-AUC, Brier, confusion matrix, and rupee cost **once** on the locked test set.
8. Store `reports/metrics.json` plus a false-positive and false-negative example.
