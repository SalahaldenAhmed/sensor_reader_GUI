import tempfile
import time
from pathlib import Path

from core.minmax import MinMaxTracker
from storage.db import init_db


def _tmp_db():
    d = Path(tempfile.mkdtemp())
    return d / "test.db"


def test_minmax_sequence_tracks_min_max_and_timestamps():
    db_path = _tmp_db()
    conn = init_db(db_path)
    t = MinMaxTracker.load_or_create(conn, "n1", "IN_TEMP")

    t.update(20.0, ts=1000)
    t.update(25.0, ts=1001)
    t.update(18.0, ts=1002)
    t.update(22.0, ts=1003)

    assert t.min_value == 18.0
    assert t.min_ts == 1002
    assert t.max_value == 25.0
    assert t.max_ts == 1001


def test_minmax_manual_reset():
    db_path = _tmp_db()
    conn = init_db(db_path)
    t = MinMaxTracker.load_or_create(conn, "n1", "IN_TEMP")
    t.update(20.0, ts=1000)
    t.update(5.0, ts=1001)
    assert t.min_value == 5.0

    t.reset(now_ts=2000)
    assert t.min_value is None
    assert t.max_value is None
    assert t.win_start_ts == 2000

    t.update(50.0, ts=2001)
    assert t.min_value == 50.0
    assert t.max_value == 50.0


def test_minmax_daily_reset():
    db_path = _tmp_db()
    conn = init_db(db_path)
    day0 = time.time()  # anchor near "now" so win_start_ts (set at creation) precedes it
    t = MinMaxTracker.load_or_create(conn, "n1", "IN_TEMP", daily_reset=True)
    t.update(10.0, ts=day0)
    t.update(30.0, ts=day0 + 10)
    assert t.min_value == 10.0
    assert t.max_value == 30.0

    next_day = day0 + 86400
    t.update(5.0, ts=next_day)
    # daily reset should have wiped the previous min/max before applying 5.0
    assert t.min_value == 5.0
    assert t.max_value == 5.0


def test_minmax_persists_across_restart():
    db_path = _tmp_db()
    conn = init_db(db_path)
    t = MinMaxTracker.load_or_create(conn, "n1", "IN_TEMP")
    t.update(20.0, ts=1000)
    t.update(5.0, ts=1001)
    t.update(40.0, ts=1002)

    # simulate restart: new connection, fresh tracker object
    conn2 = init_db(db_path)
    t2 = MinMaxTracker.load_or_create(conn2, "n1", "IN_TEMP")
    assert t2.min_value == 5.0
    assert t2.min_ts == 1001
    assert t2.max_value == 40.0
    assert t2.max_ts == 1002
