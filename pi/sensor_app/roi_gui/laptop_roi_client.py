#!/usr/bin/env python3
"""
laptop_roi_client.py — RUN THIS ON THE LAPTOP.

Half of the two-machine ROI tool. This is a thin GUI: it has no camera and
no OCR engine of its own. It connects over HTTP to pi_roi_server.py running
on the Pi, streams its live frame for you to draw on, and asks the Pi to run
OCR on a box when you hit Read -- the OCR always runs on the Pi, using the
real PaddleOCR-ONNX engine there. Nothing OCR-related needs to be installed
on the laptop.

Requires pi_roi_server.py already running on the Pi (see that file's
docstring, or roi_gui/README.md).

Controls:
  - Draw ROI                     toggle draw mode, then click-drag a box on
                                  the canvas and type a label for it
  - click a box (draw mode off)  select it (highlighted), for Delete Selected
  - Delete Selected / Clear All  remove one or all boxes
  - Read Selected / Read All     ask the Pi to run OCR on the box crop(s) of
                                  its *current* frame; shows raw_text + confidence

This does not write to config/nodes.json and does not touch anything in
sensor_app/ on the Pi side beyond calling pi_roi_server.py's HTTP API.

Run (from this machine, no OCR assets/venv from the Pi needed):
    python3 laptop_roi_client.py --host <pi-ip-or-hostname> --port 8766
See roi_gui/README.md for the full two-machine walkthrough.
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import simpledialog, messagebox
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with: pip install opencv-python numpy pillow\n"
        "If tkinter is missing on Linux: sudo apt install python3-tk"
    )

DISP_W, DISP_H = 960, 540
FRAME_POLL_S = 0.3
HTTP_TIMEOUT_S = 4
MIN_BOX_PX = 6

BG = "#101318"
SURFACE = "#171b22"
PANEL = "#1f2430"
PANEL_ALT = "#252b38"
TEXT = "#eef2f7"
MUTED = "#98a2b3"
ACCENT = "#22c55e"
BLUE = "#60a5fa"
WARN = "#fbbf24"
FLAG = "#f87171"
CARD = "#f7f9fc"
TEXT_DARK = "#18202b"
MUTED_DARK = "#7a8594"

FONT = ("Helvetica", 10)
FONT_BOLD = ("Helvetica", 10, "bold")
FONT_TITLE = ("Helvetica", 18, "bold")
FONT_SMALL = ("Helvetica", 9)
FONT_VALUE = ("Helvetica", 14, "bold")


def http_get_bytes(base_url, path, timeout=HTTP_TIMEOUT_S):
    with urllib.request.urlopen(base_url + path, timeout=timeout) as resp:
        return resp.read()


def http_get_json(base_url, path, timeout=HTTP_TIMEOUT_S):
    with urllib.request.urlopen(base_url + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class FramePoller(threading.Thread):
    """Background thread: polls GET /frame on the Pi and keeps the latest
    decoded BGR frame, same pattern as laptop/sensor_gui.py."""

    def __init__(self, base_url):
        super().__init__(daemon=True)
        self.base_url = base_url
        self._frame = None
        self._lock = threading.Lock()
        self.running = True
        self.status = "connecting..."

    def run(self):
        while self.running:
            try:
                raw = http_get_bytes(self.base_url, "/frame")
                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                    self.status = "connected"
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                self.status = f"unreachable ({e})"
            time.sleep(FRAME_POLL_S)

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self.running = False


class ROIBox:
    __slots__ = ("label", "fx", "fy", "fw", "fh", "raw_text", "confidence", "read_ts")

    def __init__(self, label, fx, fy, fw, fh):
        self.label = label
        self.fx, self.fy, self.fw, self.fh = fx, fy, fw, fh
        self.raw_text = None
        self.confidence = None
        self.read_ts = None

    def canvas_rect(self):
        return (self.fx * DISP_W, self.fy * DISP_H,
                (self.fx + self.fw) * DISP_W, (self.fy + self.fh) * DISP_H)


class ROIClientGui:
    def __init__(self, root, base_url):
        self.root = root
        self.base_url = base_url
        root.title(f"ROI Reader (client) — {base_url}")
        root.configure(bg=BG)
        root.minsize(1180, 690)

        self.boxes = []
        self.selected = None
        self.draw_mode = False
        self._drag_start = None
        self._drag_rect_id = None
        self._imgtk = None
        self._cards = {}

        self.poller = FramePoller(base_url)
        self.poller.start()

        self._build_layout()
        self._render_loop()
        self._status_loop()

    # ---- layout ----------------------------------------------------------
    def _build_layout(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=18, pady=18)
        main.grid_columnconfigure(0, weight=0)
        main.grid_columnconfigure(1, weight=1, minsize=340)
        main.grid_rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="n")

        video_panel = tk.Frame(left, bg=SURFACE, highlightbackground="#2d3441", highlightthickness=1)
        video_panel.pack(fill="x")

        head = tk.Frame(video_panel, bg=SURFACE)
        head.pack(fill="x", padx=14, pady=(12, 10))
        tk.Label(head, text="ROI Reader (Pi-connected)", bg=SURFACE, fg=TEXT, font=FONT_TITLE, anchor="w").pack(side="left")

        canvas_wrap = tk.Frame(video_panel, bg="#05070a", highlightbackground="#0b0d12", highlightthickness=1)
        canvas_wrap.pack(padx=14, pady=(0, 12))
        self.canvas = tk.Canvas(canvas_wrap, width=DISP_W, height=DISP_H, bg="black", highlightthickness=0,
                                 cursor="tcross")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        roi_bar = tk.Frame(video_panel, bg=SURFACE)
        roi_bar.pack(fill="x", padx=14, pady=(0, 8))
        self.draw_btn = self._mk_button(roi_bar, "Draw ROI: OFF", self._toggle_draw, PANEL_ALT)
        self.draw_btn.pack(side="left")
        self._mk_button(roi_bar, "Delete Selected", self._delete_selected, FLAG).pack(side="left", padx=8)
        self._mk_button(roi_bar, "Clear All", self._clear_all, FLAG).pack(side="left")

        read_bar = tk.Frame(video_panel, bg=SURFACE)
        read_bar.pack(fill="x", padx=14, pady=(0, 12))
        self._mk_button(read_bar, "Read Selected", self._read_selected, ACCENT).pack(side="left")
        self._mk_button(read_bar, "Read All", self._read_all, ACCENT).pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="")
        status_bar = tk.Frame(left, bg=SURFACE, highlightbackground="#2d3441", highlightthickness=1)
        status_bar.pack(fill="x", pady=(14, 0))
        tk.Label(status_bar, text="Status", bg=SURFACE, fg=MUTED, font=FONT_BOLD, anchor="w").pack(
            side="left", padx=(12, 8), pady=8)
        tk.Label(status_bar, textvariable=self.status_var, bg=SURFACE, fg=TEXT, font=FONT, anchor="w").pack(
            side="left", fill="x", expand=True, padx=(0, 12), pady=8)

        right = tk.Frame(main, bg=PANEL, highlightbackground="#303846", highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        rhead = tk.Frame(right, bg=PANEL)
        rhead.pack(fill="x", padx=18, pady=(16, 12))
        tk.Label(rhead, text="Boxes & Readings", bg=PANEL, fg=TEXT, font=FONT_TITLE, anchor="w").pack(side="left")
        self.count_var = tk.StringVar(value="0 boxes")
        tk.Label(rhead, textvariable=self.count_var, bg=PANEL_ALT, fg=MUTED, font=FONT_SMALL,
                 padx=10, pady=4).pack(side="right")

        scroll_area = tk.Frame(right, bg=PANEL)
        scroll_area.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.cards_canvas = tk.Canvas(scroll_area, bg=PANEL, highlightthickness=0)
        self.cards_canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(scroll_area, orient="vertical", command=self.cards_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.cards_canvas.configure(yscrollcommand=scrollbar.set)

        self.cards_frame = tk.Frame(self.cards_canvas, bg=PANEL)
        self.cards_window = self.cards_canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        self.cards_frame.bind("<Configure>", lambda e: self.cards_canvas.configure(
            scrollregion=self.cards_canvas.bbox("all")))
        self.cards_canvas.bind("<Configure>", lambda e: self.cards_canvas.itemconfigure(
            self.cards_window, width=e.width))

        self.empty_state = tk.Label(self.cards_frame, text="Draw a box on the frame to get started.",
                                     bg=PANEL, fg=MUTED, font=FONT, anchor="center", pady=40)
        self.empty_state.pack(fill="x", padx=12, pady=24)

    def _mk_button(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="#0f1117",
                          activebackground=color, activeforeground="#0f1117",
                          relief="flat", bd=0, padx=12, pady=7, font=FONT_BOLD, cursor="hand2")

    # ---- rendering ------------------------------------------------------
    def _render_loop(self):
        frame = self.poller.get()
        if frame is not None:
            disp = cv2.resize(frame, (DISP_W, DISP_H))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self._imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self._imgtk)
            for box in self.boxes:
                x1, y1, x2, y2 = box.canvas_rect()
                color = ACCENT if box is self.selected else BLUE
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                self.canvas.create_text(x1 + 3, y1 - 8, anchor="w", text=box.label, fill=color,
                                         font=("Helvetica", 10, "bold"))
        self.root.after(66, self._render_loop)

    def _status_loop(self):
        self.status_var.set(f"Pi: {self.poller.status}")
        self.root.after(500, self._status_loop)

    # ---- drawing ROIs -----------------------------------------------------
    def _toggle_draw(self):
        self.draw_mode = not self.draw_mode
        self.draw_btn.config(text=f"Draw ROI: {'ON' if self.draw_mode else 'OFF'}",
                              bg=ACCENT if self.draw_mode else PANEL_ALT)

    def _on_press(self, event):
        if self.poller.get() is None:
            return
        if self.draw_mode:
            self._drag_start = (event.x, event.y)
            self._drag_rect_id = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline=WARN, width=2)
        else:
            self._select_at(event.x, event.y)

    def _on_drag(self, event):
        if self.draw_mode and self._drag_start is not None:
            x0, y0 = self._drag_start
            self.canvas.coords(self._drag_rect_id, x0, y0, event.x, event.y)

    def _on_release(self, event):
        if not self.draw_mode or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        x0, x1 = sorted((max(0, min(x0, DISP_W)), max(0, min(x1, DISP_W))))
        y0, y1 = sorted((max(0, min(y0, DISP_H)), max(0, min(y1, DISP_H))))
        if (x1 - x0) < MIN_BOX_PX or (y1 - y0) < MIN_BOX_PX:
            return

        label = simpledialog.askstring("New ROI", "Label for this box:", parent=self.root)
        if not label:
            return
        label = label.strip()
        if any(b.label == label for b in self.boxes):
            messagebox.showwarning("New ROI", f"A box named {label!r} already exists.")
            return

        box = ROIBox(label, x0 / DISP_W, y0 / DISP_H, (x1 - x0) / DISP_W, (y1 - y0) / DISP_H)
        self.boxes.append(box)
        self.selected = box
        self._add_card(box)
        self._update_count()

    def _select_at(self, cx, cy):
        hit = None
        for box in reversed(self.boxes):
            x1, y1, x2, y2 = box.canvas_rect()
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                hit = box
                break
        self.selected = hit
        self._refresh_card_selection()

    def _delete_selected(self):
        if self.selected is None:
            return
        self._remove_box(self.selected)
        self.selected = None

    def _clear_all(self):
        if not self.boxes:
            return
        if not messagebox.askyesno("Clear All", f"Remove all {len(self.boxes)} box(es)?"):
            return
        for box in list(self.boxes):
            self._remove_box(box)
        self.selected = None

    def _remove_box(self, box):
        if box in self.boxes:
            self.boxes.remove(box)
        card = self._cards.pop(box, None)
        if card:
            card["outer"].destroy()
        self._update_count()
        if not self.boxes:
            self.empty_state.pack(fill="x", padx=12, pady=24)

    # ---- reading (over HTTP) ----------------------------------------------
    def _read_selected(self):
        if self.selected is None:
            messagebox.showinfo("Read Selected", "Click a box first (with Draw ROI off) to select it.")
            return
        self._read_boxes([self.selected])

    def _read_all(self):
        if not self.boxes:
            return
        self._read_boxes(list(self.boxes))

    def _read_boxes(self, boxes):
        for box in boxes:
            params = urllib.parse.urlencode({
                "fx": box.fx, "fy": box.fy, "fw": box.fw, "fh": box.fh, "label": box.label,
            })
            try:
                result = http_get_json(self.base_url, f"/read?{params}")
                box.raw_text = result.get("raw_text")
                box.confidence = result.get("confidence")
                box.read_ts = result.get("ts", time.time())
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                box.raw_text = f"(request failed: {e})"
                box.confidence = None
                box.read_ts = time.time()
            self._update_card(box)
        self._set_status_now(f"Read {len(boxes)} box(es)")

    def _set_status_now(self, text):
        self.status_var.set(f"Pi: {self.poller.status}  |  {text}")

    # ---- readings panel cards ----------------------------------------------
    def _add_card(self, box):
        if not self._cards:
            self.empty_state.pack_forget()

        outer = tk.Frame(self.cards_frame, bg="#dbe2ec", padx=1, pady=1)
        outer.pack(fill="x", padx=6, pady=6)
        card = tk.Frame(outer, bg=CARD)
        card.pack(fill="both", expand=True)
        card.bind("<Button-1>", lambda e, b=box: self._select_from_card(b))

        toprow = tk.Frame(card, bg=CARD)
        toprow.pack(fill="x", padx=12, pady=(10, 2))
        name_lbl = tk.Label(toprow, text=box.label, bg=CARD, fg=TEXT_DARK, font=FONT_BOLD, anchor="w")
        name_lbl.pack(side="left")

        raw_var = tk.StringVar(value="not read yet")
        raw_lbl = tk.Label(card, textvariable=raw_var, bg=CARD, fg=TEXT_DARK, font=FONT_VALUE, anchor="w")
        raw_lbl.pack(fill="x", padx=12, pady=(2, 2))

        meta_var = tk.StringVar(value="")
        meta_lbl = tk.Label(card, textvariable=meta_var, bg=CARD, fg=MUTED_DARK, font=FONT_SMALL, anchor="w")
        meta_lbl.pack(fill="x", padx=12, pady=(0, 10))

        for w in (card, toprow, name_lbl, raw_lbl, meta_lbl):
            w.bind("<Button-1>", lambda e, b=box: self._select_from_card(b))

        self._cards[box] = {"outer": outer, "raw_var": raw_var, "meta_var": meta_var}

    def _update_card(self, box):
        c = self._cards.get(box)
        if not c:
            return
        c["raw_var"].set(box.raw_text if box.raw_text else "(empty)")
        conf = f"{box.confidence:.2f}" if box.confidence is not None else "--"
        ts = time.strftime("%H:%M:%S", time.localtime(box.read_ts)) if box.read_ts else "never"
        c["meta_var"].set(f"confidence={conf}  |  read at {ts}")

    def _select_from_card(self, box):
        self.selected = box
        self._refresh_card_selection()

    def _refresh_card_selection(self):
        for b, c in self._cards.items():
            c["outer"].configure(bg=ACCENT if b is self.selected else "#dbe2ec")

    def _update_count(self):
        count = len(self.boxes)
        self.count_var.set(f"{count} {'box' if count == 1 else 'boxes'}")

    def close(self):
        self.poller.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="pi_roi_server host/IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="pi_roi_server port (default: 8766)")
    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    root = tk.Tk()
    app = ROIClientGui(root, base_url)

    def on_close():
        app.close()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
