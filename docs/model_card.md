# Model card — ReturnGuard AI (`rrs-1.4.0`)

## Intended use

Help a merchant **prioritize benign, reversible interventions** on orders that are likely to become costly returns within 30 days.

## Out of scope

Fraud execution, credential testing, payment-system bypass, automated customer punishment, protected-class inference.

## Data

Synthetic tabular orders (`synthetic-return-risk-v1`). Not production data.

## Target

`problematic_return`: returned within 30 days **and** loss-driving. A return is not automatically abuse.

## Training protocol

1. Chronological 70 / 15 / 15 split (Asia/Kolkata timestamps).
2. Preprocessing fitted on **train only**.
3. Class weights inside training.
4. Isotonic calibration on **validation only**.
5. Threshold chosen by **validation expected cost**.
6. Headline metrics computed **once** on locked test.

## Champion

Calibrated `HistGradientBoostingClassifier` vs logistic regression, a return-rate rule, and a majority baseline.

## Metrics to quote

Precision, recall, F1, PR-AUC, ROC-AUC, Brier, confusion counts, FP cost, expected decision cost. Do not lead with accuracy.

## Failure modes

- New-account orders with little history
- Category mix shift (e.g. sudden electronics share)
- Legitimate high-discount campaigns looking like risk
- Label delay in production (do not treat “no return yet” as negative if the window is open)

## Human oversight

The policy layer maps bands to **allowed** actions. Operators confirm. The model is advisory.
