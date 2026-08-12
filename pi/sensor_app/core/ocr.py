"""OCR reader interface: read_text(crop_bgr) -> (text, confidence).

Tries to load the real PaddleOCR (ONNX, PP-OCRv5 mobile rec) reader from
assets/. If the model/dict files or onnxruntime aren't available, falls back
to MockOCR so the rest of the pipeline stays testable offline.
"""
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MODEL_PATH = ASSETS_DIR / "en_PP-OCRv5_mobile_rec.onnx"
DICT_PATH = ASSETS_DIR / "ppocrv5_dict.txt"
READER_MODULE_PATH = ASSETS_DIR / "pi_ocr_reader.py"


class OCRReaderBase(ABC):
    @abstractmethod
    def read_text(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Return (raw_text, confidence in [0,1])."""
        raise NotImplementedError


class MockOCR(OCRReaderBase):
    """Returns scripted strings keyed by an arbitrary key the caller passes in
    via `script[key]`. A script value may be a constant string, or a
    list/tuple of strings that gets cycled through (one per call) -- handy for
    replaying per-frame ground truth across a sequence. Used by tests/demo
    when no real OCR model is available, and to make e2e tests deterministic.
    """

    def __init__(self, script: Optional[Dict[str, object]] = None, default: str = "0"):
        self.script = script or {}
        self.default = default
        self._cursor: Dict[str, int] = {}

    def read_text(self, crop_bgr: np.ndarray, key: Optional[str] = None) -> Tuple[str, float]:
        if key is None or key not in self.script:
            return self.default, 1.0
        entry = self.script[key]
        if isinstance(entry, (list, tuple)):
            i = self._cursor.get(key, 0) % len(entry)
            self._cursor[key] = i + 1
            return entry[i], 1.0
        return entry, 1.0


def _load_real_reader() -> Optional[OCRReaderBase]:
    if not (MODEL_PATH.exists() and DICT_PATH.exists() and READER_MODULE_PATH.exists()):
        logger.info("Real OCR assets not found; falling back to MockOCR")
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("pi_ocr_reader", READER_MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["pi_ocr_reader"] = module
        spec.loader.exec_module(module)

        underlying = module.OCRReader(model_path=str(MODEL_PATH), dict_path=str(DICT_PATH))

        class RealOCR(OCRReaderBase):
            def __init__(self, reader):
                self._reader = reader

            def read_text(self, crop_bgr: np.ndarray, key: Optional[str] = None) -> Tuple[str, float]:
                text = self._reader.read(crop_bgr)
                # The wrapped model only exposes greedy-decoded text, not a
                # softmax confidence; use presence of decoded characters as a
                # coarse confidence signal.
                confidence = 1.0 if text else 0.0
                return text, confidence

        return RealOCR(underlying)
    except Exception:
        logger.exception("Failed to load real OCR reader; falling back to MockOCR")
        return None


_real_reader = _load_real_reader()

if _real_reader is not None:
    OCR_MODE = "real"
    logger.info("OCR_MODE=real (PaddleOCR ONNX loaded)")
else:
    OCR_MODE = "mock"
    logger.info("OCR_MODE=mock (no real OCR assets / load failure)")


def get_default_reader() -> OCRReaderBase:
    """Returns the real reader if it loaded, else a default-constant MockOCR."""
    return _real_reader if _real_reader is not None else MockOCR()
