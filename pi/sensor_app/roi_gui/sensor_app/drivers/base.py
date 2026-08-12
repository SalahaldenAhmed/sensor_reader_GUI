"""Abstract camera driver interface."""
from abc import ABC, abstractmethod

import numpy as np


class CameraDriver(ABC):
    """Source of BGR frames, keyed by node_id."""

    @abstractmethod
    def get_frame(self, node_id: str) -> np.ndarray:
        """Return the next available frame (BGR ndarray) for the given node."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources."""
        raise NotImplementedError
