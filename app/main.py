from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import FRONTEND_DIR, MODEL_VERSION, POLICY_VERSION
from app.schemas import ReviewDecision, ScoreRequest
from app.services import audit
from app.services.registry import load_bundle, load_failures, load_metrics, load_quality, load_scored
from app.services.scoring import request_to_frame, score_frame

app = FastAPI(title="ReturnGuard AI", version=MODEL_VERSION)
audit.init_db()

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/api/v1/health")
def health():
    try:
        bundle = load_bundle()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": bundle.get("model_version", MODEL_VERSION),
            "policy_version": bundle.get("policy_version", POLICY_VERSION),
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={"status": "model_unavailable", "detail": str(exc), "safe_behavior": "manual_review"},
        )


@app.get("/api/v1/model-info")
def model_info():
    m = load_metrics()
    return {
        "model_version": MODEL_VERSION,
        "policy_version": POLICY_VERSION,
        "target": m.get("target"),
        "splits": m.get("splits"),
        "held_out_test": m.get("held_out_test"),
        "selected_threshold_from_validation": m.get("selected_threshold_from_validation"),
        "disclaimer": m.get("disclaimer"),
        "synthetic": True,
    }


@app.get("/api/v1/metrics")
def metrics():
    return load_metrics()


@app.get("/api/v1/quality")
def quality():
    return load_quality()


@app.get("/api/v1/drift")
def drift():
    m = load_metrics()
    return m.get("drift", {})


@app.get("/api/v1/failures")
def failures():
    return load_failures()


@app.get("/api/v1/comparison")
def comparison():
    return load_metrics().get("model_comparison_test", {})


@app.get("/api/v1/threshold-lab")
def threshold_lab():
    m = load_metrics()
    return {
        "costs": m.get("selected_threshold_from_validation"),
        "validation_choice": m.get("selected_threshold_from_validation"),
        "test_grid": m.get("test_threshold_grid", []),
        "cost_assumptions": {
            "fp_cost_inr": 120,
            "intervention_cost_inr": 80,
            "fn_fixed_inr": 220,
            "fn_margin_share": 0.28,
            "tp_save_rate": 0.55,
            "note": "Assumptions documented in docs/cost_assumptions.md. Change them to retrain, not to rewrite test labels.",
        },
    }


@app.get("/api/v1/dashboard")
def dashboard():
    df = load_scored()
    m = load_metrics()
    test = m.get("held_out_test", {})
    high = int((df["risk_band"] == "HIGH").sum()) if "risk_band" in df else int((df["risk_probability"] >= m["selected_threshold_from_validation"]["threshold"]).sum())
    review_cut = max(0.25, float(m["selected_threshold_from_validation"]["threshold"]) - 0.20)
    high_cut = float(m["selected_threshold_from_validation"]["threshold"])
    bands = pd_bands(df, high_cut, review_cut)
    recent = df.sort_values("order_time", ascending=False).head(12)
    return {
        "kpis": {
            "orders_scored": int(len(df)),
            "high_risk": int((df["risk_probability"] >= high_cut).sum()),
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "pr_auc": test.get("pr_auc"),
            "fp": test.get("fp"),
            "fn": test.get("fn"),
            "fp_cost_total": test.get("fp_cost_total"),
            "expected_decision_cost": test.get("expected_decision_cost"),
            "cost_per_1000_orders": test.get("cost_per_1000_orders"),
            "model_version": MODEL_VERSION,
        },
        "band_counts": bands,
        "category_risk": category_rates(df, high_cut),
        "recent": serialize_orders(recent, high_cut, review_cut),
        "drift": m.get("drift", {}),
        "test_locked": True,
    }


def pd_bands(df, high_cut, review_cut):
    p = df["risk_probability"]
    return {
        "LOW": int((p < review_cut).sum()),
        "REVIEW": int(((p >= review_cut) & (p < high_cut)).sum()),
        "HIGH": int((p >= high_cut).sum()),
    }


def category_rates(df, high_cut):
    g = df.groupby("product_category")["risk_probability"].mean().sort_values(ascending=False)
    return [{"category": k, "mean_risk": round(float(v) * 100, 1)} for k, v in g.items()]


def serialize_orders(df, high_cut, review_cut, include_label=True):
    rows = []
    for rec in df.to_dict(orient="records"):
        p = float(rec["risk_probability"])
        if p >= high_cut:
            band = "HIGH"
        elif p >= review_cut:
            band = "REVIEW"
        else:
            band = "LOW"
        reasons = rec.get("reason_codes")
        if hasattr(reasons, "tolist"):
            reasons = reasons.tolist()
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons.replace("'", '"'))
            except json.JSONDecodeError:
                reasons = [reasons]
        rows.append(
            {
                "order_id": rec["order_id"],
                "order_time": str(rec.get("order_time")),
                "order_value": float(rec.get("order_value", 0)),
                "product_category": rec.get("product_category"),
                "shipping_zone": rec.get("shipping_zone"),
                "risk_probability": round(p, 4),
                "risk_score": round(p * 100, 1),
                "risk_band": band,
                "reason_codes": list(reasons) if reasons is not None else [],
                "split": rec.get("split"),
                "problematic_return": int(rec["problematic_return"]) if include_label and "problematic_return" in rec else None,
                "status": rec.get("status", "pending"),
            }
        )
    return rows


@app.get("/api/v1/queue")
def queue(band: str | None = None, limit: int = 60):
    df = load_scored()
    m = load_metrics()
    high_cut = float(m["selected_threshold_from_validation"]["threshold"])
    review_cut = max(0.25, high_cut - 0.20)
    flagged = df[df["risk_probability"] >= review_cut].sort_values("risk_probability", ascending=False)
    if band:
        # apply after serialize is easier
        pass
    rows = serialize_orders(flagged.head(limit), high_cut, review_cut)
    if band:
        rows = [r for r in rows if r["risk_band"] == band.upper()]
    return {"threshold": high_cut, "items": rows}


@app.get("/api/v1/orders/{order_id}")
def order_detail(order_id: str):
    df = load_scored()
    hit = df[df["order_id"] == order_id]
    if hit.empty:
        raise HTTPException(404, "Order not found in demo scored set")
    m = load_metrics()
    high_cut = float(m["selected_threshold_from_validation"]["threshold"])
    review_cut = max(0.25, high_cut - 0.20)
    row = hit.iloc[0]
    bundle = load_bundle()
    scored = score_frame(bundle, hit)
    payload = serialize_orders(hit, high_cut, review_cut)[0]
    payload.update(scored)
    payload["feature_snapshot"] = {
        k: _jsonish(v)
        for k, v in row.to_dict().items()
        if k not in {"reason_codes"}
    }
    payload["label_available"] = True
    payload["label_note"] = "Demo scored set includes matured labels from the locked temporal test/valid slices."
    return payload


@app.post("/api/v1/score")
def score(req: ScoreRequest):
    try:
        bundle = load_bundle()
    except Exception as exc:  # noqa: BLE001
        audit.append("score_failed", req.order_id, {"error": str(exc)})
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Fail closed: route to manual review. Do not invent a score.",
        ) from exc
    try:
        df = request_to_frame(req.model_dump())
        result = score_frame(bundle, df)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit.append("prediction", req.order_id, result)
    return result


@app.post("/api/v1/review")
def review(body: ReviewDecision):
    punitive = {"cancel_customer", "blacklist", "auto_refuse_refund"}
    if body.decision in punitive:
        raise HTTPException(400, "Punitive automated actions are prohibited.")
    record = body.model_dump()
    audit.append("operator_decision", body.order_id, record)
    return {"ok": True, "stored": record, "note": "Human decision is authoritative. Model remains advisory."}


@app.get("/api/v1/audit")
def audit_log(limit: int = 80):
    return {"events": audit.list_events(limit)}


@app.get("/")
def index():
    page = FRONTEND_DIR / "index.html"
    if not page.exists():
        return {"service": "ReturnGuard AI", "docs": "/docs"}
    return FileResponse(page)


def _jsonish(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if hasattr(v, "isoformat"):
        return str(v)
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            return str(v)
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


@app.get("/{page_name}")
def spa_pages(page_name: str):
    if page_name in {"docs", "api", "openapi.json", "redoc"}:
        raise HTTPException(404)
    candidate = FRONTEND_DIR / page_name
    if candidate.exists() and candidate.suffix in {".html", ".css", ".js", ".svg", ".png"}:
        return FileResponse(candidate)
    page = FRONTEND_DIR / "index.html"
    return FileResponse(page)


def create_app():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
