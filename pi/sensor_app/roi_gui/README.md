# ROI Reader GUI

A tool for drawing ROI (region-of-interest) boxes on a camera frame, naming
them, and running the **real PaddleOCR-ONNX engine** (the same one
`pipeline.py` uses in the main app) on each box, so you can eyeball exactly
what the OCR reads before wiring anything into `config/nodes.json`.

It is intentionally simple and separate from the rest of `sensor_app/`:

- No VLM, no cloud calls, no node registration/warping, no database.
- It does not read or write `config/nodes.json` and does not touch
  `pipeline.py`, `pi_server.py`, or anything else in `sensor_app/` — the only
  thing it imports from the main app is `core/ocr.py` (read-only), to get
  the OCR engine.
- Boxes and readings live only in memory for the session — nothing is saved
  to disk. This is a look-and-tune tool, not the calibration tool
  (`setup_template.py`, one level up, still owns building the locked
  `config/nodes.json` template).

There are **two completely separate ways to run this**, as three different
scripts in this folder. Read the next section to figure out which one you
want — they are not interchangeable and need different things installed.

## Which version do I want?

**Version A — Standalone (one machine, one script: `standalone_roi_gui.py`)**

Use this if the machine you're sitting at also has (or can have) a webcam
plugged into it, directly. Camera capture, box drawing, and OCR all happen
in the same process on that one machine. This is the simplest option and
needs no networking at all.

- Running it **on your laptop** with a laptop-attached USB webcam: works,
  but the laptop needs a full local copy of the OCR assets (they're already
  in this repo under `sensor_app/assets/`, so as long as you have the repo
  checked out and its Python deps installed, this just works).
- Running it **directly on the Pi** (with a monitor+keyboard attached to the
  Pi, or over `ssh -X`): also works, same script, same camera-and-OCR-in-one-
  process model — just running on different hardware.

**Version B — Pi + Laptop split (two machines, two scripts:
`pi_roi_server.py` + `laptop_roi_client.py`)**

Use this if the camera is attached to the Pi, but you want to sit at your
laptop, look at a comfortable window, and draw boxes there — without a
monitor on the Pi and without installing the OCR engine (onnxruntime, the
ONNX model, etc.) on your laptop at all.

- `pi_roi_server.py` runs **on the Pi**. It owns the camera and the OCR
  engine, and exposes them over a small HTTP API on your LAN.
- `laptop_roi_client.py` runs **on your laptop**. It's a thin GUI with no
  camera and no OCR of its own — it streams the live frame from the Pi over
  HTTP, lets you draw/name boxes on it, and asks the Pi to run OCR on a box
  when you click Read. The actual OCR always happens on the Pi.

If you're not sure: if you can plug a USB webcam into the machine you're
sitting at, Version A is less setup. If the camera has to stay on the Pi
(e.g. it's a CSI ribbon camera, or the Pi is mounted somewhere and you don't
want to carry a monitor to it), use Version B.

---

## Version A: Standalone (`standalone_roi_gui.py`)

### 1. Setup

You need the OCR assets already in this repo
(`sensor_app/assets/en_PP-OCRv5_mobile_rec.onnx`, `assets/ppocrv5_dict.txt`,
`assets/pi_ocr_reader.py`) on the SAME machine you're running this script
on — nothing to download separately, they're checked into the repo.

Run everything from inside `sensor_app/`, using its existing venv (do not
make a second venv for this):

```bash
cd sensor_app                                   # this directory (one level above roi_gui/)
python3 -m venv venv                            # skip if the venv already exists
source venv/bin/activate
pip install -r roi_gui/requirements-standalone.txt
```

If `pip install` complains about `tkinter` — that's a system package, not a
pip package:

```bash
sudo apt install python3-tk
```

### 2. Verify the OCR engine loads for real

```bash
python3 -c "from core.ocr import OCR_MODE; print(OCR_MODE)"
```

This must print `real`. If it prints `mock`, the app still runs, but every
box will just echo back a scripted placeholder string instead of actually
reading the image — double-check the three files under `assets/` listed
above are present and that `onnxruntime` installed correctly.

### 3. Run it

```bash
cd sensor_app
source venv/bin/activate
python3 roi_gui/standalone_roi_gui.py
```

A window opens: video/image canvas on the left, box/readings list on the
right.

### 4. Using it

1. `Open Camera` (uses camera index 0 on this machine) or `Load Image...`
   (pick a `.jpg`/`.png` file, e.g. from `assets/test_set/frames/`).
2. If you opened the camera, click `Snapshot` to freeze the current frame so
   it stops moving while you draw on it. (You can reopen the camera any time
   to go live again.)
3. Click `Draw ROI` to turn drawing mode ON, then click-and-drag a rectangle
   directly on the video/image. When you release the mouse you'll be asked
   for a **label** (e.g. `IN_TEMP`, `HUMIDITY`) — just a name for the box.
4. Click `Draw ROI` again to turn drawing mode OFF, then click on a drawn
   box to select it (it highlights). `Delete Selected` removes just that
   box; `Clear All` removes every box (asks for confirmation).
5. `Read Selected` or `Read All` crops the box(es) out of whatever is
   currently displayed and runs it through the real OCR engine. The raw
   decoded text and a rough confidence score (`1.0` if any text was
   decoded, `0.0` if nothing was read) appear in the box's card on the
   right, with a timestamp.
6. Close the window (or reopen the camera) whenever — nothing is saved.

### Running over SSH (Pi with no monitor attached)

If you're driving the Pi headlessly and want to see the window on your own
screen, forward X11:

```bash
ssh -X pi@<pi-host>
cd sensor_app && source venv/bin/activate
python3 roi_gui/standalone_roi_gui.py
```

(`-X` needs an X server: built into Linux/macOS; on Windows use something
like VcXsrv or WSLg.) This still runs the whole thing — camera capture and
OCR — on the Pi; only the window is forwarded to your screen. It can feel
laggy over a slow link — if that's a problem, use Version B instead.

### Troubleshooting (Version A)

- **"Could not open camera index 0"** — no webcam attached/detected on THIS
  machine, or it's in use by another process. Use `Load Image...` instead
  to test against a file from `assets/test_set/frames/`.
- **Window doesn't appear over SSH** — you forgot `-X`/`-Y`, or no local X
  server is running.
- **Every reading is empty or garbage** — check `OCR_MODE` (step 2 above); a
  `mock` result means the ONNX assets weren't found on this machine.

---

## Version B: Pi + Laptop split (`pi_roi_server.py` + `laptop_roi_client.py`)

### Overview

```
 [ Pi: camera + OCR engine ]  <--HTTP (LAN)-->  [ Laptop: GUI window ]
   pi_roi_server.py                               laptop_roi_client.py
   GET /health, /frame, /read                      draws boxes, calls /read
```

You need BOTH scripts running at the same time, on two different machines
(or two terminals on the same machine if you're just testing this without
real hardware — see the "no camera hardware yet" note below).

### 1. On the Pi: set up and run `pi_roi_server.py`

Same venv as the rest of `sensor_app` — no GUI libraries needed here since
this script never opens a window:

```bash
cd sensor_app                                   # on the Pi
python3 -m venv venv                            # skip if it already exists
source venv/bin/activate
pip install -r roi_gui/requirements-pi-server.txt
```

Find the Pi's LAN IP address (you'll need it for the laptop side):

```bash
hostname -I
```

Start the server. If you have a real USB or CSI camera attached:

```bash
python3 roi_gui/pi_roi_server.py --host 0.0.0.0 --port 8766 --camera local:0
```

`local:0` means "local camera at index 0" — if you have more than one
camera attached, try `local:1`, etc.

**No camera hardware yet / just want to test the wiring?** Replay the
repo's test images instead, no hardware required:

```bash
python3 roi_gui/pi_roi_server.py --host 0.0.0.0 --port 8766 --camera folder
```

Leave this running. It logs each request it serves. Sanity-check it's alive
from the Pi itself:

```bash
curl http://127.0.0.1:8766/health
# -> {"ocr_mode": "real", "camera": "local:0", "frame_ready": true}
```

`ocr_mode` should say `real`. If it says `mock`, the OCR assets weren't
found — check `assets/` as described in Version A's troubleshooting.

If `frame_ready` stays `false` after a few seconds, the camera isn't
producing frames — check `--camera` is pointing at the right device and
that nothing else (another script, `pi_server.py`, a browser tab) is
already holding the camera open.

### 2. On the laptop: set up and run `laptop_roi_client.py`

This does NOT need the sensor_app venv, the OCR assets, or a checkout of
`core/` at all — it's fully independent. You can even run it from a fresh
folder with just this one file plus its requirements file copied over, or
from the full repo checkout, whichever's easier:

```bash
cd sensor_app/roi_gui                            # on the laptop
python3 -m venv venv                             # a venv just for this, if you don't already have one
source venv/bin/activate                          # Windows: venv\Scripts\activate
pip install -r requirements-laptop-client.txt
```

If `pip install` complains about `tkinter`:

```bash
sudo apt install python3-tk        # Linux
# macOS: tkinter ships with the python.org installer's Python; if you
#        installed Python via Homebrew, run: brew install python-tk
```

Run it, pointing at the Pi's IP address and the port `pi_roi_server.py` is
listening on:

```bash
python3 laptop_roi_client.py --host 192.168.1.42 --port 8766
```

(Replace `192.168.1.42` with whatever `hostname -I` printed on the Pi in
step 1. Both machines must be on the same network and able to reach each
other — if you're not sure, `ping 192.168.1.42` from the laptop first.)

A window opens showing the Pi's live camera feed, streamed over HTTP. The
status bar at bottom-left shows `Pi: connected` once it's receiving frames,
or an error like `Pi: unreachable (...)` if it can't reach the server — see
troubleshooting below.

### 3. Using it

Identical controls to Version A, except reading happens on the Pi:

1. `Draw ROI` ON, click-drag a box on the live feed, type a label when
   prompted.
2. `Draw ROI` OFF, click a box to select it. `Delete Selected` / `Clear All`
   to remove boxes.
3. `Read Selected` / `Read All` sends each box's coordinates to the Pi over
   HTTP (`GET /read?fx=...&fy=...&fw=...&fh=...&label=...`); the Pi crops
   its *current* frame, runs the real OCR engine on it, and sends back the
   raw text + confidence, which appears in that box's card exactly like in
   Version A.

Because coordinates are sent as fractions of the frame (0.0–1.0), it
doesn't matter what resolution the Pi's camera actually is — you're always
drawing "this box covers this percentage of the frame," and the Pi maps
that onto its real frame size.

### Shutting down

Ctrl-C the laptop client's terminal (or close the window) first, then
Ctrl-C `pi_roi_server.py` on the Pi. Order doesn't matter functionally, but
closing the client first avoids a few seconds of "Pi: unreachable" spam in
its terminal after the server's gone.

### Troubleshooting (Version B)

- **`Pi: unreachable (<urlopen error [Errno 111] Connection refused>)`** —
  `pi_roi_server.py` isn't running, or you've got the wrong host/port. Check
  it's still running on the Pi and that `--host 0.0.0.0` was used (not
  `127.0.0.1`, which would refuse connections from other machines).
- **`Pi: unreachable (timed out)`** — usually a network/firewall issue.
  Confirm `ping <pi-ip>` works from the laptop, and that nothing on the Pi
  (e.g. `ufw`) is blocking the port. Try `curl http://<pi-ip>:8766/health`
  from the laptop directly to isolate GUI issues from network issues.
- **Video window stays black** — the server hasn't got a frame yet
  (`frame_ready: false` in `/health`); check the camera on the Pi side per
  step 1's troubleshooting.
- **`Read` returns `(request failed: ...)`  in a box's card** — the request
  to `/read` failed (server crashed, network blip). Check the Pi terminal's
  logs for the corresponding `GET /read?...` line and any traceback above
  it.
- **Port already in use on the Pi** — something else is on 8766 (maybe a
  previous `pi_roi_server.py` you forgot was running). Either kill it or
  pick a different `--port` on both the server command and the client's
  `--port`.
- **Using the same machine for both scripts, just to test the plumbing**
  — perfectly fine: run `pi_roi_server.py --camera folder` in one terminal,
  then `laptop_roi_client.py --host 127.0.0.1 --port 8766` in another. This
  is exactly how this feature was smoke-tested during development.

---

## Quick comparison

| | Version A: `standalone_roi_gui.py` | Version B: `pi_roi_server.py` + `laptop_roi_client.py` |
|---|---|---|
| Machines needed | 1 | 2 (or 1 for testing, see above) |
| Camera location | wherever you run the script | must be attached to the Pi |
| OCR engine runs on | same machine as the GUI | the Pi, always |
| Needs OCR assets/onnxruntime on the laptop | yes (if run on laptop) | no |
| Networking required | no | yes (same LAN) |
| Best for | quick local testing, or driving the Pi directly | Pi's camera is fixed/mounted and you want a comfortable laptop window |
