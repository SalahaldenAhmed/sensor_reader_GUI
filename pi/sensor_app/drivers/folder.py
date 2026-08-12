"""Replays JPG/PNG frames from a directory. Deterministic stand-in for hardware,
used by all tests and the demo."""
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np

from drivers.base import CameraDriver

IMG_EXTS = (".jpg", ".jpeg", ".png")


class FolderDriver(CameraDriver):
    """Round-robins through sorted images in a directory (or per-node directories).

    frames_dir may be:
      - a single path/str: every node_id reads from the same folder (each node_id
        gets its own independent round-robin cursor over that folder's frames).
      - a dict[node_id, path]: each node reads from its own folder.
    """

    def __init__(self, frames_dir: Union[str, Path, Dict[str, Union[str, Path]]]):
        if isinstance(frames_dir, dict):
            self._dirs: Dict[str, Path] = {k: Path(v) for k, v in frames_dir.items()}
        else:
            self._default_dir = Path(frames_dir)
            self._dirs = {}

        self._paths_cache: Dict[str, List[Path]] = {}
        self._cursor: Dict[str, int] = {}

    def _dir_for(self, node_id: str) -> Path:
        if node_id in self._dirs:
            return self._dirs[node_id]
        if hasattr(self, "_default_dir"):
            return self._default_dir
        raise KeyError(f"No frames directory configured for node_id={node_id!r}")

    def _paths_for(self, node_id: str) -> List[Path]:
        if node_id not in self._paths_cache:
            d = self._dir_for(node_id)
            paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
            if not paths:
                raise FileNotFoundError(f"No images found in {d}")
            self._paths_cache[node_id] = paths
            self._cursor[node_id] = 0
        return self._paths_cache[node_id]

    def get_frame(self, node_id: str) -> np.ndarray:
        paths = self._paths_for(node_id)
        idx = self._cursor[node_id] % len(paths)
        frame = cv2.imread(str(paths[idx]))
        if frame is None:
            raise IOError(f"Failed to read image {paths[idx]}")
        self._cursor[node_id] = idx + 1
        return frame

    def close(self) -> None:
        pass
