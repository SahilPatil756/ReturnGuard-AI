# ReturnGuard AI

Defense-only **return-risk scoring** for online merchants. The system estimates whether a new order is likely to become a **costly / problematic return** within 30 days, explains the score, and routes a **human** to an approved, non-punitive action.

This is **not** a fraud-hacking tool. It does not generate credentials, test payments, or automate customer punishment.

> Demo data is **synthetic**. Do not describe it as real merchant traffic.

## What you get

- Temporal train / validation / **locked test** split
- Majority, rule, logistic regression, and calibrated gradient-boosting models
- Precision, recall, PR-AUC, Brier score, confusion matrix, and **false-positive rupee cost**
- Threshold lab (cost-sensitive cutoff)
- Merchant dashboard: queue, order card, audit log, data quality, model health
- Fail-closed scoring API (`POST /api/v1/score`)

## Quick start

```bash
cd returnguard-ai
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ml.models.train
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Problem definition

| Item | Definition |
| --- | --- |
| Decision time | Before fulfillment / intervention |
| Target `problematic_return` | 1 if returned within 30 days **and** the return is loss-driving (serial / high-cost pattern). Ordinary size-exchange returns are 0. |
| Features | Only information available before the decision. No post-delivery, refund-issued, or PII fields. |
| Output | Calibrated probability, 0–100 score, LOW / REVIEW / HIGH, reason codes, allowed actions |

## Cost assumptions (INR)

Documented in `docs/cost_assumptions.md`. Headline metrics use:

- False positive (unnecessary review): ₹120
- Missed costly return: ₹220 + 28% of order value
- True positive: 55% of that loss avoided, minus ₹80 intervention

Change assumptions, then **retrain**. Do not retouch the locked test labels.

## Repository

```
app/           FastAPI + scoring/policy/audit
ml/            data, features, training, evaluation
frontend/      merchant dashboard
docs/          model card, safety card, data dictionary
tests/         unit tests
```

## Demo script (about 3 minutes)

1. Start with margin destroyed by reverse logistics, not “99% accuracy.”
2. Open **Data quality** — synthetic, leakage columns absent, chronological split.
3. Open **Model health** — locked test precision/recall/PR-AUC and a real false positive.
4. Open **Threshold lab** — move the slider; watch review load vs FP cost.
5. Open a **HIGH** order — reason codes + allowed benign actions.
6. Record an operator decision — it appears in **Audit trail**.

## Safety

See `docs/safety_card.md`. Automated cancel/blacklist/refund-refusal is rejected by the API.
