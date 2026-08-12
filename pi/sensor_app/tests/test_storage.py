import json
import tempfile
import time
from pathlib import Path

from storage.cloud import StubCloudSink
from storage.csv_export import export_daily
from storage.db import (
    beat,
    fetch_latest_per_label,
    fetch_minmax,
    init_db,
    insert_reading,
    upsert_minmax,
)


def _tmp_dir():
    return Path(tempfile.mkdtemp())


def test_init_db_is_wal_mode():
    db_path = _tmp_dir() / "t.db"
    conn = init_db(db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_insert_and_fetch_readings_roundtrip():
    db_path = _tmp_dir() / "t.db"
    conn = init_db(db_path)
    now = time.time()
    insert_reading(conn, "n1", "IN_TEMP", "24.9", "C", True, True, "2492", 0.9, ts=now)
    insert_reading(conn, "n1", "IN_TEMP", "25.0", "C", True, True, "2500", 0.9, ts=now + 1)

    latest = fetch_latest_per_label(conn, "n1")
    assert len(latest) == 1
    assert latest[0]["value"] == "25.0"
    assert latest[0]["valid"] == 1


def test_upsert_minmax_roundtrip():
    db_path = _tmp_dir() / "t.db"
    conn = init_db(db_path)
    now = time.time()
    upsert_minmax(conn, "n1", "IN_TEMP", now, 18.0, now, 30.0, now + 5)
    rows = fetch_minmax(conn, "n1")
    assert len(rows) == 1
    assert rows[0]["min_value"] == 18.0
    assert rows[0]["max_value"] == 30.0

    # upsert again overwrites
    upsert_minmax(conn, "n1", "IN_TEMP", now, 10.0, now, 35.0, now + 10)
    rows = fetch_minmax(conn, "n1")
    assert len(rows) == 1
    assert rows[0]["min_value"] == 10.0
    assert rows[0]["max_value"] == 35.0


def test_beat_heartbeat():
    db_path = _tmp_dir() / "t.db"
    conn = init_db(db_path)
    beat(conn, "n1", status="ok")
    row = conn.execute("SELECT node_id, status FROM heartbeat WHERE node_id=?", ("n1",)).fetchone()
    assert row == ("n1", "ok")


def test_csv_export_writes_nonempty_file():
    db_path = _tmp_dir() / "t.db"
    conn = init_db(db_path)
    now = time.time()
    insert_reading(conn, "n1", "IN_TEMP", "24.9", "C", True, True, "2492", 0.9, ts=now)
    upsert_minmax(conn, "n1", "IN_TEMP", now, 18.0, now, 30.0, now)

    out_dir = _tmp_dir()
    date_str = time.strftime("%Y-%m-%d", time.gmtime(now))
    out_path = export_daily(db_path, out_dir, date_str)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    content = out_path.read_text()
    assert "reading" in content
    assert "minmax_summary" in content


def test_stub_cloud_sink_writes_jsonl():
    out_path = _tmp_dir() / "cloud.jsonl"
    sink = StubCloudSink(out_path)
    sink.push([{"a": 1}, {"b": 2}])
    sink.push([{"c": 3}])

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[2]) == {"c": 3}
