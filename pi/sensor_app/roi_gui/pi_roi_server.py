#!/usr/bin/env python3
"""
pi_roi_server.py — RUN THIS ON THE PI.

Half of the two-machine ROI tool. This process owns the camera and the OCR
engine. It does NOT draw anything or show a window -- it's a small headless
HTTP server (stdlib only, same pattern as pi_server.py) that:

  1. Continuously grabs frames from a camera (or replays test images) in a
     background thread, and
  2. Lets a remote client (laptop_roi_client.py) ask "what does OCR read in
     THIS rectangle of the current frame?" on demand.

Endpoints:
  GET /health                        -> {"ocr_mode": "real"|"mock", "camera": "...", "frame_ready": bool}
  GET /frame                         -> latest camera frame, JPEG bytes
  GET /read?fx=&fy=&fw=&fh=&label=   -> crops the latest frame using FRACTIONAL
                                         coords (0..1 of frame width/height),
                                         runs OCR on the crop, returns JSON:
                                         {"label": ..., "raw_text": ..., "confidence": ..., "ts": ...}

Fractional coordinates (not pixels) are used deliberately so the client
doesn't need to know this server's camera resolution ahead of time -- it
just sends "this box covers 12%..40% of frame width" and the server maps
that onto whatever frame it actually has.

This script does not touch pipeline.py, pi_server.py, config/nodes.json, or
the database -- it only imports core/ocr.py and drivers/ (read-only), so it
is completely safe to run alongside (or instead of) pi_server.py.

Run (from inside sensor_app/, with its venv active):
    python3 roi_gui/pi_roi_server.py
    python3 roi_gui/pi_roi_server.py --host 0.0.0.0 --port 8766 --camera local:0
    python3 roi_gui/pi_roi_server.py --camera folder   # no hardware camera needed, replays test images

See roi_gui/README.md for the full two-machine walkthrough.
"""
import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ocr import OCR_MODE, get_default_reader
from drivers.base import CameraDriver
from drivers.folder import FolderDriver

logger = logging.getLogger("pi_roi_server")

NODE_ID = "roi_gui"  # internal only -- this tool has no real node/config concept


def build_driver(camera_arg: str) -> CameraDriver:
    if camera_arg == "folder":
        return FolderDriver("assets/test_set/frames")
    if camera_arg.startswith("local"):
        from drivers.local import LocalCameraDriver
        idx = int(camera_arg.split(":", 1)[1]) if ":" in camera_arg else 0
        return LocalCameraDriver({NODE_ID: idx})
    raise SystemExit(f"Unknown --camera value: {camera_arg!r} (expected 'folder' or 'local[:INDEX]')")


class FrameCache:
    """Background thread that keeps grabbing frames and remembers the latest one."""

    def __init__(self, driver: CameraDriver, poll_s: float):
        self.driver = driver
        self.poll_s = poll_s
        self._lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self.driver.get_frame(NODE_ID)
                with self._lock:
                    self._frame = frame
            except Exception:
                logger.exception("Frame capture failed")
            self._stop.wait(self.poll_s)

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.driver.close()


def crop_fractional(frame: np.ndarray, fx: float, fy: float, fw: float, fh: float):
    h, w = frame.shape[:2]
    x0 = max(0, min(w, int(round(fx * w))))
    y0 = max(0, min(h, int(round(fy * h))))
    x1 = max(0, min(w, int(round((fx + fw) * w))))
    y1 = max(0, min(h, int(round((fy + fh) * h))))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def make_handler(cache: FrameCache, reader, camera_arg: str):
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

            if parsed.path == "/health":
                self._json({
                    "ocr_mode": OCR_MODE,
                    "camera": camera_arg,
                    "frame_ready": cache.latest() is not None,
                })
                return

            if parsed.path == "/frame":
                frame = cache.latest()
                if frame is None:
                    self._json({"error": "no frame yet"}, code=503)
                    return
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    self._json({"error": "jpeg encode failed"}, code=500)
                    return
                jpeg = buf.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return

            if parsed.path == "/read":
                frame = cache.latest()
                if frame is None:
                    self._json({"error": "no frame yet"}, code=503)
                    return
                try:
                    fx = float(qs["fx"][0])
                    fy = float(qs["fy"][0])
                    fw = float(qs["fw"][0])
                    fh = float(qs["fh"][0])
                except (KeyError, ValueError):
                    self._json({"error": "expected numeric fx, fy, fw, fh query params (fractions 0..1)"}, code=400)
                    return
                label = qs.get("label", [None])[0]

                crop = crop_fractional(frame, fx, fy, fw, fh)
                if crop is None or crop.size == 0:
                    self._json({"label": label, "raw_text": None, "confidence": None,
                                "ts": time.time(), "error": "empty crop"})
                    return
                raw_text, confidence = reader.read_text(crop, key=label)
                self._json({"label": label, "raw_text": raw_text, "confidence": confidence, "ts": time.time()})
                return

            self._json({"error": "not found"}, code=404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0, i.e. all interfaces)")
    parser.add_argument("--port", type=int, default=8766, help="default: 8766 (different from pi_server.py's 8765)")
    parser.add_argument("--camera", default="local:0",
                         help="'local[:INDEX]' for a real USB/CSI camera (default local:0), or 'folder' to "
                              "replay assets/test_set/frames without hardware")
    parser.add_argument("--poll-interval", type=float, default=0.1, help="seconds between frame grabs (default 0.1 = ~10fps)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    driver = build_driver(args.camera)
    cache = FrameCache(driver, args.poll_interval)
    cache.start()

    reader = get_default_reader()
    logger.info("OCR_MODE=%s", OCR_MODE)

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(cache, reader, args.camera))
    logger.info("pi_roi_server listening on http://%s:%d (camera=%s)", args.host, args.port, args.camera)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cache.stop()


if __name__ == "__main__":
    main()
