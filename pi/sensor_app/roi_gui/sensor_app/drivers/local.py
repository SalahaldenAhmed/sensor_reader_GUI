import logging
import threading
import time
import cv2

logger = logging.getLogger(__name__)


class LocalCameraDriver:
    """Persistent local USB camera driver supporting dict mapping to eliminate camera lag."""

    def __init__(self, spec):
        # spec can be a dict like {"fridge_cam": 0} or a string/int like "0"
        self.caps = {}
        self.latest_frames = {}
        self.locks = {}
        self.running = True

        # Parse mapping into node_id -> camera_index
        if isinstance(spec, dict):
            mapping = spec
        else:
            try:
                mapping = {"default": int(spec)}
            except (ValueError, TypeError):
                mapping = {"default": 0}

        # Open each unique camera device and start background reader threads
        for key, idx in mapping.items():
            try:
                cam_idx = int(idx)
            except (ValueError, TypeError):
                cam_idx = 0

            logger.info("Initializing camera index %d for key '%s'...", cam_idx, key)
            cap = cv2.VideoCapture(cam_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.caps[key] = cap
            self.latest_frames[key] = None
            self.locks[key] = threading.Lock()

            # Start continuous streaming thread for each camera
            thread = threading.Thread(
                target=self._reader_loop, args=(key,), daemon=True
            )
            thread.start()

    def _reader_loop(self, key):
        cap = self.caps[key]
        lock = self.locks[key]
        while self.running:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    with lock:
                        self.latest_frames[key] = frame
            time.sleep(0.01)  # ~30 FPS continuous read

    def get_frame(self, node_id=None):
        # Fallback to first available camera if node_id not found
        key = node_id if node_id in self.locks else next(iter(self.locks), None)
        if key is None:
            return None

        with self.locks[key]:
            if self.latest_frames[key] is not None:
                return self.latest_frames[key].copy()
        return None

    def close(self):
        self.running = False
        for cap in self.caps.values():
            if cap.isOpened():
                cap.release()
