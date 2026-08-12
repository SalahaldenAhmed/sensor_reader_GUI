"""HTTP camera driver for ESP32-CAM style devices. Stub/untested: must import
cleanly without `requests` actually being reachable; raises only on get_frame()."""
from typing import Dict

import cv2
import numpy as np

from drivers.base import CameraDriver

try:
    import requests
except ImportError:  # pragma: no cover - requests is an optional dependency here
    requests = None


class ESP32Driver(CameraDriver):
    """Fetches a JPEG snapshot via HTTP GET {base_url}/capture per node."""

    def __init__(self, base_url_by_node: Dict[str, str], timeout: float = 5.0):
        self._base_url_by_node = base_url_by_node
        self._timeout = timeout

    def get_frame(self, node_id: str) -> np.ndarray:
        if requests is None:
            raise RuntimeError("requests package not available; cannot fetch ESP32 frame")
        base_url = self._base_url_by_node.get(node_id)
        if not base_url:
            raise KeyError(f"No base_url configured for node_id={node_id!r}")
        resp = requests.get(f"{base_url}/capture", timeout=self._timeout)
        resp.raise_for_status()
        data = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise IOError(f"Failed to decode JPEG from {base_url}/capture")
        return frame

    def close(self) -> None:
        pass
