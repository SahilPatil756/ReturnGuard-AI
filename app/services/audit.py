from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import ROOT

DB_PATH = ROOT / "data" / "processed" / "audit.sqlite"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = _connect()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            order_id TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    con.commit()
    con.close()


def append(event_type: str, order_id: str | None, payload: dict):
    init_db()
    con = _connect()
    con.execute(
        "INSERT INTO audit_events (ts, event_type, order_id, payload) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), event_type, order_id, json.dumps(payload, default=str)),
    )
    con.commit()
    con.close()


def list_events(limit: int = 80) -> list[dict]:
    init_db()
    con = _connect()
    rows = con.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    out = []
    for r in rows:
        item = dict(r)
        item["payload"] = json.loads(item["payload"])
        out.append(item)
    return out
