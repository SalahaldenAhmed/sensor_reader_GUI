"""Per (node, label) min/max tracking over a resettable window.

Default window is "since boot" (i.e. since the tracker/window was last reset).
Supports a daily reset (call maybe_daily_reset(now_ts) each cycle) and a manual
reset(). State is persisted through storage.db so it survives a restart.
"""
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

from storage.db import upsert_minmax


def _day_start_ts(ts: float) -> float:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return day_start.timestamp()


class MinMaxTracker:
    """Tracks min/max for one (node_id, label) pair, backed by SQLite."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        node_id: str,
        label: str,
        win_start_ts: Optional[float] = None,
        daily_reset: bool = False,
    ):
        self.conn = conn
        self.node_id = node_id
        self.label = label
        self.daily_reset = daily_reset
        self.win_start_ts = win_start_ts if win_start_ts is not None else time.time()
        self.min_value: Optional[float] = None
        self.min_ts: Optional[float] = None
        self.max_value: Optional[float] = None
        self.max_ts: Optional[float] = None

    @classmethod
    def load_or_create(cls, conn: sqlite3.Connection, node_id: str, label: str, daily_reset: bool = False) -> "MinMaxTracker":
        row = conn.execute(
            "SELECT win_start_ts, min_value, min_ts, max_value, max_ts FROM minmax WHERE node_id=? AND label=?",
            (node_id, label),
        ).fetchone()
        tracker = cls(conn, node_id, label, daily_reset=daily_reset)
        if row:
            win_start_ts, min_value, min_ts, max_value, max_ts = row
            tracker.win_start_ts = win_start_ts
            tracker.min_value = min_value
            tracker.min_ts = min_ts
            tracker.max_value = max_value
            tracker.max_ts = max_ts
        else:
            tracker._persist()
        return tracker

    def _persist(self) -> None:
        upsert_minmax(
            self.conn, self.node_id, self.label, self.win_start_ts,
            self.min_value, self.min_ts, self.max_value, self.max_ts,
        )

    def maybe_daily_reset(self, now_ts: Optional[float] = None) -> bool:
        if not self.daily_reset:
            return False
        now_ts = now_ts if now_ts is not None else time.time()
        if _day_start_ts(now_ts) > _day_start_ts(self.win_start_ts):
            self.reset(now_ts)
            return True
        return False

    def reset(self, now_ts: Optional[float] = None) -> None:
        self.win_start_ts = now_ts if now_ts is not None else time.time()
        self.min_value = None
        self.min_ts = None
        self.max_value = None
        self.max_ts = None
        self._persist()

    def update(self, value: float, ts: Optional[float] = None) -> None:
        ts = ts if ts is not None else time.time()
        self.maybe_daily_reset(ts)
        if self.min_value is None or value < self.min_value:
            self.min_value = value
            self.min_ts = ts
        if self.max_value is None or value > self.max_value:
            self.max_value = value
            self.max_ts = ts
        self._persist()
