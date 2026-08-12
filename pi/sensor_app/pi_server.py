"""HTTP server that runs the sensor_app pipeline continuously and exposes
readings + the live frame + health over a small stdlib HTTP API, for
sensor_gui.py (or any other client) to poll over the LAN.

Endpoints:
  GET /health              -> {"nodes": {node_id: {"status": "ok"|"degraded"|"error"|"starting", "last_seen_ts": float|null}}}
  GET /readings            -> {"nodes": {node_id: {"boxes": {label: {value, unit, valid, confidence, raw_text, x, y, w, h, ts}}}}}
  GET /readings?node=NODE  -> same, filtered to one node
  GET /frame               -> latest raw camera frame for the first configured node, as JPEG bytes
  GET /frame?node=NODE     -> latest raw camera frame for that node, as JPEG bytes

Notes:
  - The box x/y/w/h returned by /readings are the reference-frame coordinates
    from config/nodes.json (pre-ORB-warp). They're accurate for the reference
    image the node was calibrated against and are meant for a client to draw
    an approximate overlay, not as ground truth per-frame geometry.
  - By default this replays assets/test_set frames via FolderDriver -- a
    deterministic stand-in for a camera, useful for local testing and for
    verifying the GUI without hardware. Pass --camera local[:INDEX] to use a
    real USB/CSI camera via LocalCameraDriver instead (untested against real
    hardware -- see README "Known limitations").

Run (from inside sensor_app/):
    python3 pi_server.py
    python3 pi_server.py --host 0.0.0.0 --port 8765 --camera local:0
"""
import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from core.nodes import NodeStore
from core.ocr import OCR_MODE, get_default_reader
from drivers.base import CameraDriver
from drivers.folder import FolderDriver
from pipeline import run_once
from storage.db import fetch_latest_per_label, init_db

logger = logging.getLogger("pi_server")


class CachingDriver(CameraDriver):
    """Wraps another driver and remembers the last frame served per node_id,
    so /frame can serve exactly what the pipeline just processed without
    triggering a second, separate capture."""

    def __init__(self, inner: CameraDriver):
        self._inner = inner
        self._lock = threading.Lock()
        self._latest = {}

    def get_frame(self, node_id: str):
        frame = self._inner.get_frame(node_id)
        with self._lock:
            self._latest[node_id] = frame
        return frame

    def latest(self, node_id: str):
        with self._lock:
            frame = self._latest.get(node_id)
        return None if frame is None else frame.copy()

    def close(self) -> None:
        self._inner.close()


class PipelineState:
    """Lock-protected view of the running pipeline for the HTTP handlers."""

    def __init__(self, store: NodeStore, driver: CachingDriver, db):
        self.store = store
        self.driver = driver
        self.db = db
        self._lock = threading.Lock()
        self._status = {n.node_id: {"status": "starting", "last_seen_ts": None} for n in store}

    def note(self, node_id, status, ts):
        with self._lock:
            self._status[node_id] = {"status": status, "last_seen_ts": ts}

    def health(self):
        with self._lock:
            return {"nodes": dict(self._status)}

    def readings(self, node_id=None):
        node_ids = [node_id] if node_id else list(self.store.nodes.keys())
        out = {}
        for nid in node_ids:
            if nid not in self.store.nodes:
                continue
            node = self.store.get(nid)
            latest_by_label = {r["label"]: r for r in fetch_latest_per_label(self.db, nid)}
            boxes = {}
            for box in node.boxes:
                r = latest_by_label.get(box.label)
                boxes[box.label] = {
                    "value": r["value"] if r else None,
                    "unit": box.unit,
                    "valid": bool(r["valid"]) if r else False,
                    "confidence": r["confidence"] if r else None,
                    "raw_text": r["raw_text"] if r else None,
                    "ts": r["ts"] if r else None,
                    "x": box.x, "y": box.y, "w": box.w, "h": box.h,
                }
            out[nid] = {"boxes": boxes}
        return {"nodes": out}

    def frame_jpeg(self, node_id):
        frame = self.driver.latest(node_id)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None


def _pipeline_loop(state: PipelineState, ocr, cadence_s: float, stop_event: threading.Event):
    while not stop_event.is_set():
        for node in state.store:
            try:
                run_once(node, state.driver, ocr, state.db)
                status = "ok" if (node.last_health or {}).get("ok", True) else "degraded"
            except Exception:
                logger.exception("run_once failed for node %s", node.node_id)
                status = "error"
            state.note(node.node_id, status, time.time())
        stop_event.wait(cadence_s)


def make_handler(state: PipelineState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            node_id = qs.get("node", [None])[0]

            if parsed.path == "/health":
                self._json(state.health())
            elif parsed.path == "/readings":
                self._json(state.readings(node_id))
            elif parsed.path == "/frame":
                nid = node_id or next(iter(state.store.nodes.keys()))
                jpeg = state.frame_jpeg(nid)
                if jpeg is None:
                    self._json({"error": f"no frame yet for node {nid!r}"}, code=503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            else:
                self._json({"error": "not found"}, code=404)

    return Handler


def build_driver(camera_arg: str, store: NodeStore) -> CameraDriver:
    if camera_arg == "folder":
        return FolderDriver({n.node_id: n.driver_spec["frames_dir"] for n in store})
    if camera_arg.startswith("local"):
        from drivers.local import LocalCameraDriver
        idx = int(camera_arg.split(":", 1)[1]) if ":" in camera_arg else 0
        return LocalCameraDriver({n.node_id: idx for n in store})
    raise SystemExit(f"Unknown --camera value: {camera_arg!r} (expected 'folder' or 'local[:INDEX]')")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default="config/nodes.json")
    parser.add_argument("--cadence", type=float, default=1.0, help="seconds between pipeline passes")
    parser.add_argument("--db", default="pi_server.db")
    parser.add_argument("--camera", default="folder",
                         help="'folder' (default, replays assets/test_set) or 'local[:INDEX]' for a real USB/CSI camera")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    store = NodeStore(args.config)
    driver = CachingDriver(build_driver(args.camera, store))
    ocr = get_default_reader()
    logger.info("OCR_MODE=%s", OCR_MODE)
    db = init_db(args.db)

    state = PipelineState(store, driver, db)
    stop_event = threading.Event()
    worker = threading.Thread(target=_pipeline_loop, args=(state, ocr, args.cadence, stop_event), daemon=True)
    worker.start()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    logger.info("pi_server listening on http://%s:%d (nodes: %s)",
                args.host, args.port, ", ".join(store.nodes.keys()))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        worker.join(timeout=2)
        driver.close()


if __name__ == "__main__":
    main()
