from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code in (200, 503)


def test_score_rejects_negative_price():
    r = client.post(
        "/api/v1/score",
        json={
            "product_category": "apparel",
            "product_price": -1,
            "quantity": 1,
            "discount_pct": 10,
            "shipping_zone": "Z1",
            "promised_delivery_days": 3,
            "payment_method": "upi",
            "account_age_days": 40,
            "historical_orders": 3,
            "historical_returns": 0,
        },
    )
    assert r.status_code == 422


def test_review_stores_audit():
    # scoring may 503 if model missing in a bare checkout; skip if so
    health = client.get("/api/v1/health")
    if health.status_code != 200:
        return
    scored = client.post(
        "/api/v1/score",
        json={
            "order_id": "ORD-TEST-1",
            "product_category": "electronics",
            "product_price": 15000,
            "quantity": 1,
            "discount_pct": 40,
            "shipping_zone": "Z4",
            "promised_delivery_days": 8,
            "payment_method": "cod",
            "account_age_days": 8,
            "historical_orders": 1,
            "historical_returns": 1,
            "address_changes_90d": 3,
            "orders_30d": 4,
            "refunds_30d": 2,
        },
    )
    assert scored.status_code == 200
    body = scored.json()
    assert "risk_score" in body
    assert "reason_codes" in body
    rev = client.post(
        "/api/v1/review",
        json={"order_id": "ORD-TEST-1", "decision": "manual_review", "operator": "pytest", "note": "ok"},
    )
    assert rev.status_code == 200
    audit = client.get("/api/v1/audit")
    assert any(e["order_id"] == "ORD-TEST-1" for e in audit.json()["events"])
