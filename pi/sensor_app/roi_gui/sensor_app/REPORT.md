# Overnight build report — sensor_app

Built autonomously overnight per the brief: manual-ROI + ORB-registration +
PaddleOCR runtime path. No YOLO/VLM. All work is local to `sensor_app/`.

## Modules built

| Module | Status |
|---|---|
| `drivers/base.py`, `folder.py`, `local.py`, `esp32.py` | Done. `FolderDriver` (round-robin, per-node dirs) is what all tests/demo use. `local.py`/`esp32.py` import cleanly but are untested stubs (no camera/ESP32 hardware here). |
| `core/registration.py` | Done. ORB(nfeatures=800) + BFMatcher(NORM_HAMMING) + Lowe ratio test + `estimateAffinePartial2D` (RANSAC). Verified it recovers a known ~25px translate + 1.5° rotation to within ~8px on a real test frame, and correctly reports `health.ok=False` on blank/scrambled frames. |
| `core/ocr.py` | Done. Tries to load the real PaddleOCR ONNX reader from `assets/`; falls back to `MockOCR` otherwise. **`OCR_MODE = "real"` in this environment** — the asset files and onnxruntime were present, so the real model loaded successfully. |
| `core/reading.py` | Done. `parse_reference`, `parse_device_range`, `read_anchored` ported **verbatim** (semantics unchanged, verified against the documented test cases). Added `TemporalVoter` (K=5 ring buffer, N=3 agreement, exact match for clocks / epsilon for numbers) on top. |
| `core/minmax.py` | Done. `MinMaxTracker`, SQLite-backed, supports manual `reset()` and `daily_reset` (UTC day-boundary). Verified persistence across a simulated restart (new connection + reload). |
| `core/nodes.py` | Done. `Box`/`Node` dataclasses + `NodeStore` that loads `config/nodes.json` and builds each node's ORB reference from its reference image at load time. |
| `storage/db.py` | Done. SQLite WAL (`PRAGMA journal_mode=WAL`), `readings`/`minmax`/`heartbeat` tables, short autocommit-per-call transactions. |
| `storage/csv_export.py` | Done. `export_daily` writes per-day readings + a min/max summary row per sensor. |
| `storage/cloud.py` | Done. `CloudSink` ABC + `StubCloudSink` (JSONL append) — proves the push contract, no real Supabase credentials anywhere. |
| `config/nodes.json` | Authored: two nodes, `htc2_cam` (IN_TEMP, OUT_TEMP, Humidity in C/%RH, Time clock) and `fridge_cam` (FRIDGE_TEMP, range -40..40C). |
| `setup_template.py` | Done. Manual ROI + heuristic (`%`→humidity, `:`→clock, else temp/C) template builder, writes/updates `config/nodes.json`. No VLM. |
| `pipeline.py` | Done. `run_once` (capture → register/warp → per-box crop/OCR/anchor/vote → minmax → insert_reading → beat) and `run_loop` (cadence-driven). Registration failure or out-of-bounds box → hold last-good, flag stale/invalid, never raises. |
| `run_demo.py` | Done, runs cleanly end-to-end (see below). |

## Test results

```
33 passed in 3.33s
```

All of `test_registration.py`, `test_reading.py`, `test_voting.py`,
`test_minmax.py`, `test_storage.py`, `test_pipeline_e2e.py` are green,
including the MockOCR e2e (always runs) and a bonus real-OCR informational
pass over the fridge frames (only runs when `OCR_MODE == "real"`).

Run with: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -v`
(Note: plain `pytest`/`python3 -m pytest` without that env var fails to start
in this environment — a stray `anyio` pytest plugin in `~/.local` is
incompatible with the installed pytest 6.2.5 and breaks plugin autoload. Not
a sensor_app bug; worth fixing in the venv at some point — see "Needs
attention".)

## OCR_MODE: real (and why)

`assets/pi_ocr_reader.py`, `assets/en_PP-OCRv5_mobile_rec.onnx`,
`assets/ppocrv5_dict.txt` were all present, and `onnxruntime` is installed in
the venv, so `core/ocr.py` loaded the real PP-OCRv5 mobile rec ONNX model.
Spot-checked against `assets/test_set/crops/`, e.g.:
`frame_001_Humidity.jpg` → `'48%'`, `frame_001_Time.jpg` → `'801'` — correct.

The wrapped model only exposes greedy CTC-decoded text via `OCRReader.read()`,
not a softmax confidence score, so `core/ocr.py`'s `RealOCR.read_text()` uses
a coarse heuristic confidence (1.0 if any text decoded, else 0.0). A real
confidence score would need re-implementing the CTC decode against the raw
logits instead of calling `.read()` — left as a next step, not attempted
tonight per the 15-minute stub rule.

### Real-OCR raw outputs observed

On the pre-cropped `assets/test_set/crops/*.jpg` images (correctly framed to
the digit display), the real OCR reads cleanly, e.g. `'48%'`, `'801'`.

On the **full, uncropped** `assets/test_set/fridge_thermometers/*.jpg` images
(1500×1500, whole-scene photos — there's no calibrated ROI for these yet),
the real OCR reads garbage as expected, since it's looking at a full scene
instead of a tight digit crop:

- `fridge_thermometer_001.jpg`: raw_text=`'1 '`
- `fridge_thermometer_002.jpg`: raw_text=`'a'`
- `fridge_thermometer_003.jpg`: raw_text=`'O'`

This is **expected and informational only** (per the brief, this pass never
hard-fails on accuracy) — it confirms the OCR plumbing works, not that the
ROIs are calibrated. Same root cause as the `run_demo.py` output below.

### `run_demo.py` output

`python3 run_demo.py` runs 10 cycles across both nodes against real OCR with
**zero exceptions**. Because `config/nodes.json`'s pixel rects are
placeholder/illustrative (not calibrated against an actual live frame — see
"Needs attention" below), the crops don't land on real digits, so OCR reads
garbage and `read_anchored` correctly rejects it (digit-count/range checks
fail) — `valid=False` throughout, values stay `None` since nothing ever
votes stable. `fridge_cam` also intermittently shows `health_ok=False`
because the registration health gate (correctly) rejects the
`fridge_thermometer_003.jpg` frame, which has too few inlier ORB matches
against the reference (also observed directly in
`test_e2e_handles_registration_failure_gracefully`). This demonstrates the
pipeline's failure handling works correctly end-to-end; it does not
demonstrate accurate readings, because the ROIs were never calibrated against
a live frame.

The MockOCR e2e test (`test_pipeline_e2e.py`), by contrast, *does* show
correct, meaningful values — it bypasses the pixel-rect problem entirely
since it returns scripted ground-truth strings from `labels.csv` regardless
of what's cropped, proving the reading/voting/minmax/storage chain is
correct independent of ROI calibration.

## Needs attention (stubbed / blocked, <15min rule)

1. **`config/nodes.json` pixel ROIs are placeholders, not calibrated.** I
   authored plausible-looking rects without an interactive tool to click ROIs
   on a live frame (out of scope for an unattended overnight run). Real OCR
   therefore reads garbage in `run_demo.py`. The reading/voting/storage logic
   is fully tested and correct via `MockOCR`; only the geometry needs fixing.
2. **`RealOCR` confidence is a heuristic** (1.0/0.0 based on non-empty
   output), not a true softmax-based score, since `assets/pi_ocr_reader.py`'s
   public API only returns decoded text.
3. **`drivers/local.py` and `drivers/esp32.py` are untested** — no camera or
   ESP32 hardware available in this environment. They import cleanly and have
   reasonable error handling, but have never executed against real hardware.
4. **pytest plugin autoload bug** in the venv (see Test results) — works
   around it with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, didn't dig further since
   it's an environment issue, not a sensor_app bug.

## Next steps for the morning

1. **Calibrate real pixel ROIs**: grab a live frame from each camera, use
   `setup_template.py`'s `build_node_template`/`write_node_to_config` (or a
   quick interactive cv2 ROI picker) to replace the placeholder rects in
   `config/nodes.json` with real ones, sampling actual on-screen text for
   `sample_text` so `parse_reference` derives correct decimals/int_digits.
2. **Re-run `run_demo.py` with calibrated ROIs** and confirm real OCR
   produces sane, voting-stable values (the MockOCR e2e proves the rest of
   the chain already works).
3. **Swap `FolderDriver` for the real camera driver** once hardware is
   available (`LocalCameraDriver` for a USB/CSI cam, or `ESP32Driver` for the
   HTTP-based cam) — both are written and import cleanly but untested.
4. **Wire `StubCloudSink` to real Supabase** once credentials are available
   (same `push(rows)` interface, just swap the implementation).
5. Consider replacing `RealOCR`'s heuristic confidence with a true CTC
   softmax-based score (would need to bypass `OCRReader.read()` and decode
   the raw onnxruntime logits directly) if confidence-gating turns out to
   matter in practice.

---
*Note: `test_pipeline_e2e.py::test_e2e_real_ocr_fridge_pass_is_informational`
appends a raw-output block to this file every time the suite runs (by
design, per the brief). Re-running `pytest` will add another copy below this
line — safe to ignore/trim.*

### Real-OCR informational pass over fridge_thermometers (from test_pipeline_e2e.py)

- fridge_thermometer_001.jpg: raw_text='1 ' confidence=1.0
- fridge_thermometer_002.jpg: raw_text='a' confidence=1.0
- fridge_thermometer_003.jpg: raw_text='O' confidence=1.0

### Real-OCR informational pass over fridge_thermometers (from test_pipeline_e2e.py)

- fridge_thermometer_001.jpg: raw_text='1 ' confidence=1.0
- fridge_thermometer_002.jpg: raw_text='a' confidence=1.0
- fridge_thermometer_003.jpg: raw_text='O' confidence=1.0

### Real-OCR informational pass over fridge_thermometers (from test_pipeline_e2e.py)

- fridge_thermometer_001.jpg: raw_text='1 ' confidence=1.0
- fridge_thermometer_002.jpg: raw_text='a' confidence=1.0
- fridge_thermometer_003.jpg: raw_text='O' confidence=1.0

### Real-OCR informational pass over fridge_thermometers (from test_pipeline_e2e.py)

- fridge_thermometer_001.jpg: raw_text='1 ' confidence=1.0
- fridge_thermometer_002.jpg: raw_text='a' confidence=1.0
- fridge_thermometer_003.jpg: raw_text='O' confidence=1.0
