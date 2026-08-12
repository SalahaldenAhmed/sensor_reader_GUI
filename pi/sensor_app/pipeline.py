"""Wires registration -> OCR -> reading -> voting -> storage together.

run_once: one capture+process cycle for a single node. Never raises on
registration failure, OCR failure, or out-of-range readings -- those are all
flagged in the stored reading instead, and last-good is held.
run_loop: drives run_once across all nodes on a cadence.
"""
import logging
import time
from typing import List, Optional

import numpy as np

from core.nodes import Node
from core.reading import read_anchored
from core.registration import register_and_warp
from storage.db import beat, insert_reading

logger = logging.getLogger(__name__)


def _crop(frame: np.ndarray, warped_box: dict) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x0 = max(0, int(round(warped_box["x"])))
    y0 = max(0, int(round(warped_box["y"])))
    x1 = min(w, int(round(warped_box["x"] + warped_box["w"])))
    y1 = min(h, int(round(warped_box["y"] + warped_box["h"])))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def run_once(node: Node, drivers, ocr, db, cloud=None) -> List[dict]:
    """Runs one capture+process cycle for `node`. Returns the list of result
    dicts that were written to the readings table (also pushed to `cloud` if
    given)."""
    now = time.time()
    results: List[dict] = []

    try:
        frame = drivers.get_frame(node.node_id)
    except Exception:
        logger.exception("Failed to get frame for node %s", node.node_id)
        for box in node.boxes:
            results.append(_hold_last_good(node, box, db, now, reason="frame capture failed"))
        beat(db, node.node_id, status="error", ts=now)
        if cloud:
            cloud.push(results)
        return results

    box_dicts = [b.to_box_dict() for b in node.boxes]
    warped_boxes, health = register_and_warp(frame, node.reference, box_dicts)
    node.last_health = health

    if not health["ok"]:
        for box in node.boxes:
            results.append(_hold_last_good(node, box, db, now, reason=f"registration: {health['reason']}"))
        beat(db, node.node_id, status="degraded", ts=now)
        if cloud:
            cloud.push(results)
        return results

    warped_by_label = {wb["label"]: wb for wb in warped_boxes}

    for box in node.boxes:
        wb = warped_by_label[box.label]
        if not wb["in_bounds"]:
            results.append(_hold_last_good(node, box, db, now, reason="box out of frame bounds"))
            continue

        crop = _crop(frame, wb)
        if crop is None or crop.size == 0:
            results.append(_hold_last_good(node, box, db, now, reason="empty crop"))
            continue

        raw_text, confidence = ocr.read_text(crop, key=box.label)
        field = box.to_field_dict()
        value_str, ocr_valid, reason = read_anchored(raw_text, field)
        box.last_good = field.get("last_good", box.last_good)

        voter = node.voter_for(box.label)
        if ocr_valid:
            stable_value, is_stable = voter.commit(value_str)
            published_valid = bool(is_stable)
        else:
            # Don't feed a rejected/out-of-range reading into the voter's
            # buffer -- hold whatever was last published stable.
            stable_value = voter._stable_value
            published_valid = False

        if published_valid and box.type not in ("clock", "time") and stable_value is not None:
            try:
                node.minmax_for(db, box.label).update(float(stable_value), ts=now)
            except (TypeError, ValueError):
                pass

        row = dict(
            ts=now, node_id=node.node_id, label=box.label,
            value=stable_value, unit=box.unit, valid=published_valid,
            health_ok=True, raw_text=raw_text, confidence=confidence,
        )
        insert_reading(
            db, node.node_id, box.label, stable_value, box.unit,
            published_valid, True, raw_text, confidence, ts=now,
        )
        results.append(row)

    beat(db, node.node_id, status="ok", ts=now)
    if cloud:
        cloud.push(results)
    return results


def _hold_last_good(node: Node, box, db, now: float, reason: str) -> dict:
    voter = node.voter_for(box.label)
    stable_value = getattr(voter, "_stable_value", None)
    row = dict(
        ts=now, node_id=node.node_id, label=box.label,
        value=stable_value, unit=box.unit, valid=False,
        health_ok=False, raw_text=None, confidence=None, reason=reason,
    )
    insert_reading(
        db, node.node_id, box.label, stable_value, box.unit,
        False, False, None, None, ts=now,
    )
    return row


def run_loop(nodes, drivers, ocr, db, cloud=None, cadence_s: float = 1.0, n_iterations: Optional[int] = None):
    """Iterates over all nodes repeatedly, sleeping `cadence_s` between full
    passes. If n_iterations is given, stops after that many passes (used by
    tests/demo); otherwise runs forever."""
    i = 0
    while n_iterations is None or i < n_iterations:
        for node in nodes:
            run_once(node, drivers, ocr, db, cloud)
        i += 1
        if n_iterations is None or i < n_iterations:
            time.sleep(cadence_s)
