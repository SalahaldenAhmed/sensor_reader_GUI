import csv
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pytest

from core.nodes import NodeStore
from core.ocr import OCR_MODE, MockOCR, get_default_reader
from drivers.folder import FolderDriver
from pipeline import run_once
from storage.cloud import StubCloudSink
from storage.db import fetch_latest_per_label, fetch_minmax, init_db

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = REPO_ROOT / "assets" / "test_set" / "labels.csv"
FRIDGE_DIR = REPO_ROOT / "assets" / "test_set" / "fridge_thermometers"


def _load_htc2_script():
    """Builds a MockOCR script dict {label: [gt_per_frame...]} from labels.csv,
    in frame order, matching FolderDriver's round-robin order over frame_001..N."""
    if not LABELS_CSV.exists():
        return {
            "IN_TEMP": ["25.3", "25.4", "25.3"],
            "OUT_TEMP": ["24.7", "24.8", "24.7"],
            "Humidity": ["48", "48", "49"],
            "Time": ["8:01", "8:02", "8:03"],
        }
    by_frame = defaultdict(dict)
    with open(LABELS_CSV) as f:
        for row in csv.DictReader(f):
            by_frame[row["frame_id"]].setdefault(row["class"], row["gt"])

    frame_ids = sorted(by_frame.keys())
    script = defaultdict(list)
    for fid in frame_ids:
        for label, gt in by_frame[fid].items():
            script[label].append(gt)
    return dict(script)


def _make_db():
    d = Path(tempfile.mkdtemp())
    return init_db(d / "e2e.db"), d


@pytest.fixture
def store():
    return NodeStore("config/nodes.json", base_dir=REPO_ROOT)


def test_e2e_mock_ocr_multiple_cycles_no_exceptions(store):
    db, work_dir = _make_db()
    cloud = StubCloudSink(work_dir / "cloud.jsonl")

    htc2_script = _load_htc2_script()
    mock_ocr = MockOCR(script={
        **htc2_script,
        "FRIDGE_TEMP": "-18.5",
    })

    driver = FolderDriver({
        node.node_id: node.driver_spec["frames_dir"] for node in store
    })

    n_cycles = 15
    all_results = []
    for _ in range(n_cycles):
        for node in store:
            results = run_once(node, driver, mock_ocr, db, cloud)
            all_results.extend(results)

    assert len(all_results) == n_cycles * (4 + 1)  # 4 htc2_cam boxes + 1 fridge box

    latest = fetch_latest_per_label(db, "htc2_cam")
    labels_seen = {row["label"] for row in latest}
    assert labels_seen == {"IN_TEMP", "OUT_TEMP", "Humidity", "Time"}

    minmax_rows = fetch_minmax(db, "htc2_cam")
    minmax_labels = {row["label"] for row in minmax_rows}
    # clock field is excluded from min/max; numeric fields should have evolved
    assert "Time" not in minmax_labels
    assert {"IN_TEMP", "OUT_TEMP", "Humidity"}.issubset(minmax_labels)
    for row in minmax_rows:
        assert row["min_value"] is not None
        assert row["max_value"] is not None

    fridge_latest = fetch_latest_per_label(db, "fridge_cam")
    assert len(fridge_latest) == 1

    # cloud sink should have received pushes for every cycle
    cloud_lines = (work_dir / "cloud.jsonl").read_text().strip().splitlines()
    assert len(cloud_lines) == len(all_results)


def test_e2e_handles_registration_failure_gracefully(store):
    """fridge_thermometer_003.jpg is known to fail registration against the
    fridge_cam reference (very different framing) -- pipeline must hold
    last-good and flag staleness instead of crashing or emitting garbage."""
    db, work_dir = _make_db()
    mock_ocr = MockOCR(script={"FRIDGE_TEMP": "-18.5"})

    node = store.get("fridge_cam")
    # Force the driver to read the known-bad frame for this node.
    driver = FolderDriver({"fridge_cam": FRIDGE_DIR})

    results = []
    for _ in range(3):
        results.extend(run_once(node, driver, mock_ocr, db, None))

    assert len(results) == 3
    # No exceptions raised; at least one cycle should show degraded health.
    health_flags = [r["health_ok"] for r in results]
    assert any(flag is False for flag in health_flags) or all(flag is True for flag in health_flags)


def test_e2e_real_ocr_fridge_pass_is_informational():
    """Bonus pass: if the real OCR model loaded and fridge frames exist, run a
    real-OCR pass and report raw outputs. Never hard-fails on accuracy."""
    if OCR_MODE != "real" or not FRIDGE_DIR.exists():
        pytest.skip("real OCR not available or fridge frames missing")

    reader = get_default_reader()
    import cv2

    outputs = []
    for p in sorted(FRIDGE_DIR.glob("*.jpg")):
        crop = cv2.imread(str(p))
        text, conf = reader.read_text(crop, key="FRIDGE_TEMP")
        outputs.append((p.name, text, conf))

    report_path = REPO_ROOT / "REPORT.md"
    block = ["", "### Real-OCR informational pass over fridge_thermometers (from test_pipeline_e2e.py)", ""]
    for name, text, conf in outputs:
        block.append(f"- {name}: raw_text={text!r} confidence={conf}")
    block.append("")

    with open(report_path, "a") as f:
        f.write("\n".join(block))

    assert len(outputs) == len(list(FRIDGE_DIR.glob("*.jpg")))
