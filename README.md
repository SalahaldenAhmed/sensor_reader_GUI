# Sensor Reader

A vision-based system for reading analog/digital sensor displays (temperature,
humidity, clock-style readouts) off a camera feed. A Raspberry Pi captures
frames, registers them against a reference image to correct for camera drift,
crops out the configured display regions, runs OCR, and votes across frames
for a stable value. A laptop GUI polls the Pi over the LAN and shows the live
feed plus the current readings.

## Current status / known limitations — read this first

**The pixel ROIs (regions of interest) in `pi/sensor_app/config/nodes.json`
are placeholders.** They were authored by hand without ever being checked
against a real frame from an actual camera. As shipped, every reading will
come back `value=None valid=False` — this is expected, not a bug. You need to
calibrate the ROIs against your own camera before you'll see real numbers.
See **"Calibrating your ROIs"** below; it's the first thing to do after
getting the system running.

Other things to know before you dig in:

- `pi/sensor_app/drivers/local.py` (USB/CSI camera) and `esp32.py`
  (HTTP-based ESP32 camera) are **untested** — they import cleanly and have
  reasonable error handling, but have never run against real hardware. The
  default driver (`FolderDriver`, replaying a folder of JPGs) is what all 33
  tests and the demo actually exercise.
- `RealOCR`'s confidence score (in `core/ocr.py`) is a coarse heuristic —
  `1.0` if any text decoded, `0.0` if not — not a true softmax/CTC
  confidence. The wrapped OCR model only exposes greedy-decoded text.
- The `MockOCR` path (used in most of the test suite) *does* produce correct
  values end-to-end and proves the registration → reading → voting → storage
  chain is correct, independent of ROI calibration or real-OCR accuracy.
- The old GUI used to let you drag boxes live on the video feed and push them
  to the Pi ad hoc. That workflow doesn't exist anymore — see "What changed
  from the old GUI" below.

Full build notes and test rationale are in `pi/sensor_app/REPORT.md`.

## Architecture

```
   Raspberry Pi (pi/)                          Laptop (laptop/)
  ┌───────────────────────────┐               ┌───────────────────────┐
  │ camera / FolderDriver      │               │                       │
  │        │                   │               │     sensor_gui.py     │
  │        ▼                   │   HTTP, LAN   │   (tkinter, polls     │
  │  ORB registration/warp     │◄──────────────┤    /health /readings  │
  │        │                   │               │    /frame every       │
  │        ▼                   │               │    ~0.3-1s)           │
  │  per-box crop → PaddleOCR  │               │                       │
  │        │                   │               │  live view + per-box  │
  │        ▼                   │               │  reading cards        │
  │  anchored parse → vote     │               │                       │
  │        │                   │               └───────────────────────┘
  │        ▼                   │
  │  SQLite (readings/minmax)  │
  │        │                   │
  │        ▼                   │
  │  pi_server.py (stdlib      │
  │  http.server): serves      │
  │  /health /readings /frame  │
  └───────────────────────────┘
```

- `pi/sensor_app/pipeline.py` — the capture → register → crop → OCR → parse →
  vote → store loop (`run_once` / `run_loop`). Never raises on a bad
  frame/box/OCR result; holds the last-good value and flags the reading
  invalid/stale instead.
- `pi/sensor_app/pi_server.py` — new in this release. Runs the pipeline
  continuously in a background thread and exposes it over plain HTTP so a
  client (the GUI, or anything else) doesn't need to embed the pipeline
  itself.
- `laptop/sensor_gui.py` — a pure display client. It has no OCR/registration
  code of its own; it just polls the Pi's HTTP endpoints and renders what
  comes back.

### What changed from the old GUI

The previous `sensor_gui.py` talked to a different, VLM-based Pi server
(`live_dual_vlm_server.py`, Florence/Gemini/Qwen) over raw pickle sockets, and
let you **draw boxes live on the video feed and push them to the Pi** as an ad
hoc template (`"Send Boxes to Pi"`). That workflow is gone:

- The VLM path (Florence, Gemini, Qwen, `"VLM Auto-Setup"`) has been removed
  entirely. The Pi always runs the PaddleOCR-ONNX pipeline in `sensor_app/`.
- Boxes are no longer pushed from the GUI at runtime. They come from the Pi's
  `config/nodes.json`, calibrated ahead of time with `setup_template.py` (see
  below). The GUI just displays whatever boxes the Pi reports, read-only, with
  a green/red outline for valid/invalid.
- The GUI now talks HTTP + JSON (`urllib`, stdlib) instead of a bespoke pickle
  socket protocol, and the Pi's address is a `--host`/`--port` CLI flag
  instead of hardcoded.

Reconciling live box-drawing with ORB-registration-based coordinate mapping
would be a real feature (mapping GUI-drawn display pixels back through the
registration warp into `config/nodes.json`), not a protocol swap — it wasn't
attempted here. If you want that back, `setup_template.py` is the place to
start: it already builds box entries from `(label, x, y, w, h, sample_text)`
tuples, it just currently expects those coordinates from a script/notebook,
not a live drag gesture.

## Install

Assumes Ubuntu (or another Debian-based Linux) and Python 3.10+. The Pi and
laptop can use different Python versions as long as both are 3.9+.

### Pi setup

```bash
cd pi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pi.txt
```

`onnxruntime` on a Raspberry Pi may need the ARM-specific wheel; if
`pip install onnxruntime==1.23.2` fails to find a wheel for your Pi's
architecture, check the [onnxruntime releases page](https://github.com/microsoft/onnxruntime/releases)
for an ARM build matching your Python version.

### Laptop setup

```bash
cd laptop
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-laptop.txt
```

If `import tkinter` fails:

```bash
sudo apt install python3-tk
```

## Running the Pi server

```bash
cd pi/sensor_app
python3 pi_server.py --host 0.0.0.0 --port 8765
```

By default this replays the sample frames under `assets/test_set/` via
`FolderDriver` — no camera needed, useful to confirm the whole chain works
before touching hardware. To use a real USB/CSI camera:

```bash
python3 pi_server.py --host 0.0.0.0 --port 8765 --camera local:0
```

(`0` is the `/dev/video0` index; `drivers/local.py` is untested against real
hardware — see "Current status" above.)

Find the Pi's IP with `hostname -I` or `ip addr`.

Optional: to auto-start on boot, create a systemd unit, e.g.
`/etc/systemd/system/sensor-pi-server.service`:

```ini
[Unit]
Description=Sensor reader Pi server
After=network.target

[Service]
WorkingDirectory=/home/pi/sensor_reader_release/pi/sensor_app
ExecStart=/home/pi/sensor_reader_release/pi/venv/bin/python3 pi_server.py --host 0.0.0.0 --port 8765
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

then `sudo systemctl enable --now sensor-pi-server`.

## Running the laptop GUI

```bash
cd laptop
python3 sensor_gui.py --host <pi-ip-or-hostname> --port 8765
```

You should see a window with the Pi's live camera feed on the left (boxes
overlaid — red outline while the ROIs are uncalibrated, since nothing reads
as valid yet) and a "Readings" panel on the right with one card per
node/box, showing the last value, unit, and freshness. Use the "Node"
dropdown if the Pi has more than one node configured (the sample config has
two: `htc2_cam` and `fridge_cam`).

To test locally without a Pi at all, run `pi_server.py` on the same machine
(`--host 127.0.0.1`) and point the GUI at `--host 127.0.0.1` (the GUI's
default).

## Running the headless pipeline (no GUI, no server)

```bash
cd pi/sensor_app
python3 run_demo.py
```

Runs 10 cycles across both configured nodes against the sample frames,
printing each reading, then a min/max summary and a CSV export path. Useful
for a quick sanity check of the pipeline in isolation.

## Running tests

```bash
cd pi/sensor_app
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -v
```

The `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required in some environments: a
stray `anyio` pytest plugin installed elsewhere on the system (e.g. in
`~/.local`) is incompatible with the pinned `pytest==6.2.5` and breaks
plugin autoload, causing `pytest` to fail before it even collects tests. This
isn't a bug in this project — it's an environment conflict — and disabling
plugin autoload works around it without disabling any of `sensor_app`'s own
tests. If `plain pytest` works fine in your environment, you can drop the env
var.

Expect `33 passed`.

## Configuration

`pi/sensor_app/config/nodes.json` defines one entry per **node** (a physical
camera position). Each node has:

- `node_id` — unique name, used in the API and GUI.
- `driver_spec` — how to get frames (`{"type": "folder", "frames_dir": ...}`
  for `FolderDriver`; adapt for `local`/`esp32` drivers).
- `reference_image` — a path to a frame that defines the "home" camera pose
  ORB registration warps every subsequent frame back to.
- `boxes` — a list of regions to read, each with `label`, pixel `x/y/w/h`
  (relative to the reference image), `type` (`temp`/`humidity`/`clock`),
  `unit`, `decimals`, `int_digits`, `range_min`/`range_max` (for
  sanity-checking OCR output), and `ref_value`/`last_good` (seed values).

### Calibrating your ROIs

This is the step that turns `value=None valid=False` into real numbers.

1. Grab a real frame from your camera and save it, e.g.
   `assets/my_cam/frame_001.jpg` (this becomes your `reference_image`).
2. Open it in any image viewer that shows pixel coordinates (or `cv2` — e.g.
   `python3 -c "import cv2; im=cv2.imread('assets/my_cam/frame_001.jpg'); cv2.imshow('x', im); cv2.setMouseCallback('x', lambda *a: print(a[1], a[2])); cv2.waitKey(0)"`
   and click the corners of each display region) to find the pixel
   `x, y, w, h` of each region you want to read, and note what text is
   actually showing there (e.g. `"25.3C"`, `"48%"`).
3. Use `setup_template.py` as a library to build and write the node:

   ```python
   import sys; sys.path.insert(0, ".")  # run from pi/sensor_app/
   from setup_template import build_node_template, write_node_to_config

   node_cfg = build_node_template(
       node_id="my_cam",
       reference_image_path="assets/my_cam/frame_001.jpg",
       driver_spec={"type": "folder", "frames_dir": "assets/my_cam"},
       manual_boxes=[
           {"label": "IN_TEMP", "x": 120, "y": 340, "w": 160, "h": 80, "sample_text": "25.3C"},
           {"label": "Humidity", "x": 320, "y": 340, "w": 140, "h": 80, "sample_text": "48%"},
       ],
   )
   write_node_to_config(node_cfg, "config/nodes.json")
   ```

   `sample_text` matters: it's how `parse_reference` infers the number of
   integer digits and decimal places to expect, and how the humidity/clock
   type heuristics kick in (`%` → humidity, `:` → clock, else temperature).

4. Re-run `python3 run_demo.py` or restart `pi_server.py` and confirm the new
   node reads real, stable values instead of `None`.
5. If registration keeps reporting `health.ok=False`, the reference image
   probably doesn't have enough distinctive visual texture for ORB to match
   against (e.g. a mostly blank/glare-heavy frame) — pick a reference frame
   with clear, well-lit detail.
