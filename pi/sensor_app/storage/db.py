"""SQLite storage, WAL mode, short transactions for concurrent-friendly access."""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    node_id TEXT NOT NULL,
    label TEXT NOT NULL,
    value TEXT,
    unit TEXT,
    valid INTEGER NOT NULL,
    health_ok INTEGER NOT NULL,
    raw_text TEXT,
    confidence REAL
);

CREATE INDEX IF NOT EXISTS idx_readings_node_label_ts ON readings(node_id, label, ts);

CREATE TABLE IF NOT EXISTS minmax (
    node_id TEXT NOT NULL,
    label TEXT NOT NULL,
    win_start_ts REAL NOT NULL,
    min_value REAL,
    min_ts REAL,
    max_value REAL,
    max_ts REAL,
    PRIMARY KEY (node_id, label)
);

CREATE TABLE IF NOT EXISTS heartbeat (
    node_id TEXT PRIMARY KEY,
    last_seen_ts REAL NOT NULL,
    status TEXT NOT NULL
);
"""


def init_db(path: Union[str, Path]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def _txn(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def insert_reading(
    conn: sqlite3.Connection,
    node_id: str,
    label: str,
    value: Optional[str],
    unit: Optional[str],
    valid: bool,
    health_ok: bool,
    raw_text: Optional[str],
    confidence: Optional[float],
    ts: Optional[float] = None,
) -> int:
    ts = ts if ts is not None else time.time()
    with _txn(conn):
        cur = conn.execute(
            "INSERT INTO readings (ts, node_id, label, value, unit, valid, health_ok, raw_text, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, node_id, label, value, unit, int(valid), int(health_ok), raw_text, confidence),
        )
        return cur.lastrowid


def upsert_minmax(
    conn: sqlite3.Connection,
    node_id: str,
    label: str,
    win_start_ts: float,
    min_value: Optional[float],
    min_ts: Optional[float],
    max_value: Optional[float],
    max_ts: Optional[float],
) -> None:
    with _txn(conn):
        conn.execute(
            """
            INSERT INTO minmax (node_id, label, win_start_ts, min_value, min_ts, max_value, max_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id, label) DO UPDATE SET
                win_start_ts=excluded.win_start_ts,
                min_value=excluded.min_value,
                min_ts=excluded.min_ts,
                max_value=excluded.max_value,
                max_ts=excluded.max_ts
            """,
            (node_id, label, win_start_ts, min_value, min_ts, max_value, max_ts),
        )


def beat(conn: sqlite3.Connection, node_id: str, status: str = "ok", ts: Optional[float] = None) -> None:
    ts = ts if ts is not None else time.time()
    with _txn(conn):
        conn.execute(
            """
            INSERT INTO heartbeat (node_id, last_seen_ts, status)
            VALUES (?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET last_seen_ts=excluded.last_seen_ts, status=excluded.status
            """,
            (node_id, ts, status),
        )


def fetch_latest_per_label(conn: sqlite3.Connection, node_id: Optional[str] = None) -> List[Dict]:
    where = "WHERE node_id = ?" if node_id else ""
    params = (node_id,) if node_id else ()
    rows = conn.execute(
        f"""
        SELECT r.* FROM readings r
        INNER JOIN (
            SELECT node_id, label, MAX(ts) AS max_ts FROM readings
            {where}
            GROUP BY node_id, label
        ) latest
        ON r.node_id = latest.node_id AND r.label = latest.label AND r.ts = latest.max_ts
        """,
        params,
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM readings LIMIT 0").description]
    return [dict(zip(cols, row)) for row in rows]


def fetch_minmax(conn: sqlite3.Connection, node_id: Optional[str] = None) -> List[Dict]:
    if node_id:
        rows = conn.execute("SELECT * FROM minmax WHERE node_id = ?", (node_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM minmax").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM minmax LIMIT 0").description]
    return [dict(zip(cols, row)) for row in rows]
