"""Export a day's readings + min/max summary to CSV."""
import csv
import sqlite3
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Union


def _day_bounds_ts(date: Union[str, datetime]) -> "tuple[float, float]":
    if isinstance(date, str):
        d = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        d = date.replace(tzinfo=date.tzinfo or timezone.utc)
    start = datetime.combine(d.date(), dtime.min, tzinfo=d.tzinfo)
    end = datetime.combine(d.date(), dtime.max, tzinfo=d.tzinfo)
    return start.timestamp(), end.timestamp()


def export_daily(db_path: Union[str, Path], out_dir: Union[str, Path], date: Union[str, datetime]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
    out_path = out_dir / f"readings_{date_str}.csv"

    start_ts, end_ts = _day_bounds_ts(date)
    conn = sqlite3.connect(str(db_path))
    try:
        readings = conn.execute(
            "SELECT ts, node_id, label, value, unit, valid, health_ok, raw_text, confidence "
            "FROM readings WHERE ts >= ? AND ts <= ? ORDER BY ts",
            (start_ts, end_ts),
        ).fetchall()
        minmax_rows = conn.execute(
            "SELECT node_id, label, win_start_ts, min_value, min_ts, max_value, max_ts FROM minmax"
        ).fetchall()
    finally:
        conn.close()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_type", "ts", "node_id", "label", "value", "unit",
                          "valid", "health_ok", "raw_text", "confidence"])
        for ts, node_id, label, value, unit, valid, health_ok, raw_text, confidence in readings:
            writer.writerow(["reading", ts, node_id, label, value, unit, valid, health_ok, raw_text, confidence])
        for node_id, label, win_start_ts, min_value, min_ts, max_value, max_ts in minmax_rows:
            writer.writerow([
                "minmax_summary", win_start_ts, node_id, label,
                f"min={min_value}@{min_ts} max={max_value}@{max_ts}", "", "", "", "", "",
            ])

    return out_path
