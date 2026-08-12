"""cv2.VideoCapture-backed camera driver. Stub-safe: importable with no camera
attached; raises only when get_frame() is actually called without hardware."""
from typing import Dict, Optional

import cv2
import numpy as np

from drivers.base import CameraDriver


class LocalCameraDriver(CameraDriver):
    """Wraps cv2.VideoCapture, one capture per node_id."""

    def __init__(self, device_index_by_node: Optional[Dict[str, int]] = None):
        self._device_index_by_node = device_index_by_node or {}
        self._caps: Dict[str, "cv2.VideoCapture"] = {}

    def _cap_for(self, node_id: str) -> "cv2.VideoCapture":
        if node_id not in self._caps:
            idx = self._device_index_by_node.get(node_id, 0)
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open local camera index {idx} for node {node_id!r}")
            self._caps[node_id] = cap
        return self._caps[node_id]

    def get_frame(self, node_id: str) -> np.ndarray:
        cap = self._cap_for(node_id)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise IOError(f"Failed to read frame from local camera for node {node_id!r}")
        return frame

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
