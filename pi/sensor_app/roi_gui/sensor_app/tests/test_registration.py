from pathlib import Path

import cv2
import numpy as np

from core.registration import build_reference, register_and_warp

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_FRAME = REPO_ROOT / "assets" / "test_set" / "frames" / "frame_001.jpg"


def _load_or_generate_reference_frame() -> np.ndarray:
    if TEST_FRAME.exists():
        frame = cv2.imread(str(TEST_FRAME))
        if frame is not None:
            return frame

    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (600, 440), (10, 10, 10), -1)
    for _ in range(400):
        x, y = rng.integers(40, 600), rng.integers(40, 440)
        c = int(rng.integers(0, 255))
        cv2.circle(frame, (int(x), int(y)), int(rng.integers(2, 6)), (c, c, c), -1)
    return frame


def _known_transform(shape) -> np.ndarray:
    h, w = shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 1.5, 1.0)
    M[:, 2] += [25, 18]
    return M


def test_register_and_warp_recovers_known_transform():
    ref_frame = _load_or_generate_reference_frame()
    h, w = ref_frame.shape[:2]
    reference = build_reference(ref_frame)

    box = {
        "label": "DISPLAY", "x": w * 0.3, "y": h * 0.3, "w": w * 0.15, "h": h * 0.1,
        "type": "temp", "unit": "C", "decimals": 1, "int_digits": 2,
        "range_min": -50, "range_max": 100, "ref_value": 20.0, "last_good": None,
    }

    M = _known_transform(ref_frame.shape)
    frame = cv2.warpAffine(ref_frame, M, (w, h))

    warped_boxes, health = register_and_warp(frame, reference, [box])

    assert health["ok"] is True
    assert health["inliers"] >= 10
    assert health["inlier_ratio"] >= 0.3

    wb = warped_boxes[0]

    corner = np.array([box["x"], box["y"], 1.0])
    expected = M @ corner

    assert abs(wb["x"] - expected[0]) < 10
    assert abs(wb["y"] - expected[1]) < 10
    assert wb["in_bounds"] is True


def test_register_and_warp_blank_frame_fails_health():
    ref_frame = _load_or_generate_reference_frame()
    h, w = ref_frame.shape[:2]
    reference = build_reference(ref_frame)

    box = {
        "label": "DISPLAY", "x": w * 0.3, "y": h * 0.3, "w": w * 0.15, "h": h * 0.1,
        "type": "temp", "unit": "C", "decimals": 1, "int_digits": 2,
        "range_min": -50, "range_max": 100, "ref_value": 20.0, "last_good": None,
    }

    blank = np.zeros((h, w, 3), dtype=np.uint8)
    warped_boxes, health = register_and_warp(blank, reference, [box])

    assert health["ok"] is False
    assert all(wb["in_bounds"] is False for wb in warped_boxes)


def test_register_and_warp_scrambled_frame_fails_health():
    ref_frame = _load_or_generate_reference_frame()
    h, w = ref_frame.shape[:2]
    reference = build_reference(ref_frame)

    rng = np.random.default_rng(7)
    scrambled = ref_frame.copy()
    flat = scrambled.reshape(-1, 3)
    rng.shuffle(flat)
    scrambled = flat.reshape(ref_frame.shape)

    box = {
        "label": "DISPLAY", "x": w * 0.3, "y": h * 0.3, "w": w * 0.15, "h": h * 0.1,
        "type": "temp", "unit": "C", "decimals": 1, "int_digits": 2,
        "range_min": -50, "range_max": 100, "ref_value": 20.0, "last_good": None,
    }

    warped_boxes, health = register_and_warp(scrambled, reference, [box])
    assert health["ok"] is False
