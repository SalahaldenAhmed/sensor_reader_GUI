"""ORB + partial-affine frame registration.

Strategy: ROIs are locked once at setup time against a reference frame. Every
runtime frame is registered back to that reference via ORB feature matching and
a 4-DOF partial affine (translation + rotation + uniform scale), then every
locked box is warped by that single estimated transform so it tracks the
(shaking/sagging) camera mount.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MIN_MATCHES = 15
MIN_INLIERS = 10
MIN_INLIER_RATIO = 0.3
LOWE_RATIO = 0.75


def _orb():
    return cv2.ORB_create(nfeatures=800)


def build_reference(frame_bgr: np.ndarray, context_mask: Optional[np.ndarray] = None) -> dict:
    """Build a reference descriptor set from the setup frame.

    context_mask, if given, is a uint8 mask (255 = keep) that should exclude the
    digit ROIs so keypoints come from stable surrounding scenery rather than the
    digits themselves (which change frame to frame).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    orb = _orb()
    keypoints, descriptors = orb.detectAndCompute(gray, context_mask)
    return {
        "gray": gray,
        "shape": frame_bgr.shape[:2],
        "keypoints": keypoints,
        "descriptors": descriptors,
    }


def _box_corners(box: dict) -> np.ndarray:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


def _warp_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts_h = pts.reshape(-1, 1, 2)
    warped = cv2.transform(pts_h, M)
    return warped.reshape(-1, 2)


def _in_bounds(corners: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
    h, w = frame_shape
    xs, ys = corners[:, 0], corners[:, 1]
    return bool(xs.min() >= 0 and ys.min() >= 0 and xs.max() <= w and ys.max() <= h)


def register_and_warp(
    frame_bgr: np.ndarray, reference: dict, boxes: List[dict]
) -> Tuple[List[dict], dict]:
    """Register frame_bgr against the stored reference and warp all boxes by the
    single recovered transform.

    Returns (warped_boxes, health). warped_boxes is a list of dicts (copies of
    the input boxes) with corners/warped x,y,w,h (axis-aligned bounding box of
    the warped quad) and an `in_bounds` flag added. health describes transform
    quality; when health["ok"] is False, warped_boxes still contains best-effort
    geometry but callers should treat all boxes as unreadable this frame.
    """
    ref_desc = reference.get("descriptors")
    ref_kp = reference.get("keypoints")

    def _fail(reason: str, n_matches: int = 0) -> Tuple[List[dict], dict]:
        out = []
        for b in boxes:
            wb = dict(b)
            wb["in_bounds"] = False
            wb["warped_corners"] = _box_corners(b).tolist()
            out.append(wb)
        return out, {
            "ok": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "n_matches": n_matches,
            "reason": reason,
        }

    if ref_desc is None or ref_kp is None or len(ref_kp) == 0:
        return _fail("empty reference descriptors")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    orb = _orb()
    kp, desc = orb.detectAndCompute(gray, None)
    if desc is None or len(kp) == 0:
        return _fail("no keypoints in frame")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc, ref_desc, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    if len(good) < MIN_MATCHES:
        return _fail(f"too few good matches ({len(good)})", len(good))

    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inlier_mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)

    if M is None:
        return _fail("estimateAffinePartial2D failed", len(good))

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = inliers / len(good) if good else 0.0

    ok = inliers >= MIN_INLIERS and inlier_ratio >= MIN_INLIER_RATIO
    reason = "" if ok else f"insufficient inliers ({inliers}/{len(good)})"

    # M maps frame(src) -> reference(dst). Boxes are stored in reference coords,
    # so we need the inverse transform: reference -> current frame.
    M3 = np.vstack([M, [0, 0, 1]]).astype(np.float64)
    M_inv3 = np.linalg.inv(M3)
    M_inv = M_inv3[:2, :]

    frame_shape = frame_bgr.shape[:2]
    warped_boxes = []
    for b in boxes:
        corners = _box_corners(b)
        warped_corners = _warp_points(corners, M_inv) if ok else corners
        wb = dict(b)
        xs, ys = warped_corners[:, 0], warped_corners[:, 1]
        wb["warped_corners"] = warped_corners.tolist()
        wb["x"] = float(xs.min())
        wb["y"] = float(ys.min())
        wb["w"] = float(xs.max() - xs.min())
        wb["h"] = float(ys.max() - ys.min())
        wb["in_bounds"] = ok and _in_bounds(warped_corners, frame_shape)
        warped_boxes.append(wb)

    health = {
        "ok": ok,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "n_matches": len(good),
        "reason": reason,
    }
    return warped_boxes, health
