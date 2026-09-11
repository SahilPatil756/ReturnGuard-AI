# Cost assumptions

These values are **planning assumptions** for the demo, not accounting truth.

| Event | INR treatment |
| --- | --- |
| True negative (routine order, no flag) | ₹0 |
| False positive (flag a normal order) | ₹120 review labor + friction |
| Intervention on a true positive | ₹80 (message / confirmation) |
| Missed costly return (false negative) | ₹220 reverse-logistics overhead + 28% of order value (margin + restock) |
| Caught costly return (true positive) | Save 55% of the FN loss, minus the ₹80 intervention |

Expected decision cost on a batch:

`FP × 120 + Σ_FN (220 + 0.28 × value) − Σ_TP (0.55 × (220 + 0.28 × value) − 80)`

Sensitivity: rerun `python -m ml.models.train` after editing `app/config.py`.
