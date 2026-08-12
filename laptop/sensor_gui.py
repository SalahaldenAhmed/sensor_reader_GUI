#!/usr/bin/env python3
"""
sensor_gui.py — Laptop-side single-window GUI for the sensor reader.

Talks to pi_server.py (sensor_app/pi_server.py) over plain HTTP:
    GET /health              -> per-node pipeline status
    GET /readings[?node=ID]  -> latest value/unit/valid/confidence per box
    GET /frame[?node=ID]     -> latest raw camera frame, JPEG bytes

This is a pure display client. It does not run any OCR/registration itself
and does not push box definitions to the Pi — the boxes it shows are exactly
whatever nodes/boxes are configured in the Pi's sensor_app/config/nodes.json
(see "Calibrating your ROIs" in the README to give those boxes real
coordinates). There is no VLM path anymore; the Pi always runs the
PaddleOCR-ONNX pipeline.

Requirements on the LAPTOP:
    pip install opencv-python numpy pillow
    # tkinter is usually built in; on Debian/Ubuntu if missing:
    #   sudo apt install python3-tk

Run:
    python3 sensor_gui.py --host <pi-ip-or-hostname> --port 8765
    (defaults to 127.0.0.1:8765, i.e. a pi_server running on the same
    machine you're launching the GUI from -- handy for local testing)
"""
import argparse
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with: pip install opencv-python numpy pillow\n"
        "If tkinter is missing on Linux: sudo apt install python3-tk"
    )

DISP_W, DISP_H = 640, 360   # display size the video canvas is drawn at
FRAME_POLL_S = 0.8
READINGS_POLL_S = 1.0
HTTP_TIMEOUT_S = 4
STALE_AFTER_SECONDS = 12

# --- colors / theme ---
BG = "#101318"
SURFACE = "#171b22"
PANEL = "#1f2430"
PANEL_ALT = "#252b38"
CARD = "#f7f9fc"
CARD_STALE = "#edf1f6"
CARD_BORDER = "#dbe2ec"
CARD_BORDER_FRESH = "#89d69f"
CARD_BORDER_INVALID = "#f0b9b9"
ACCENT = "#22c55e"
ACCENT_DIM = "#7b8794"
TEXT = "#eef2f7"
TEXT_DARK = "#18202b"
TEXT_SOFT = "#536173"
MUTED = "#98a2b3"
MUTED_DARK = "#7a8594"
BLUE = "#60a5fa"
WARN = "#fbbf24"
FLAG = "#f87171"

FONT = ("Helvetica", 10)
FONT_BOLD = ("Helvetica", 10, "bold")
FONT_TITLE = ("Helvetica", 18, "bold")
FONT_SECTION = ("Helvetica", 12, "bold")
FONT_VALUE = ("Helvetica", 28, "bold")
FONT_SMALL = ("Helvetica", 9)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_json(base_url, path, timeout=HTTP_TIMEOUT_S):
    with urllib.request.urlopen(base_url + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_bytes(base_url, path, timeout=HTTP_TIMEOUT_S):
    with urllib.request.urlopen(base_url + path, timeout=timeout) as resp:
        return resp.read()


class ReadingsPoller(threading.Thread):
    """Background thread: polls GET /readings on a fixed cadence and keeps
    the most recent parsed JSON for the GUI to render."""

    def __init__(self, base_url):
        super().__init__(daemon=True)
        self.base_url = base_url
        self.latest = {"nodes": {}}
        self.lock = threading.Lock()
        self.running = True
        self.status = "connecting..."

    def run(self):
        while self.running:
            try:
                data = http_get_json(self.base_url, "/readings")
                with self.lock:
                    self.latest = data
                self.status = "connected"
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                self.status = f"readings unreachable ({e})"
            time.sleep(READINGS_POLL_S)

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False


class FramePoller(threading.Thread):
    """Background thread: polls GET /frame?node=<id> for whichever node is
    currently selected and keeps the most recent decoded BGR frame."""

    def __init__(self, base_url, node_getter):
        super().__init__(daemon=True)
        self.base_url = base_url
        self.node_getter = node_getter
        self.latest = None
        self.lock = threading.Lock()
        self.running = True
        self.status = "connecting..."

    def run(self):
        while self.running:
            node = self.node_getter()
            path = "/frame" if not node else f"/frame?node={node}"
            try:
                raw = http_get_bytes(self.base_url, path)
                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        self.latest = frame
                    self.status = "stream connected"
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                self.status = f"frame unreachable ({e})"
                time.sleep(1)
            time.sleep(FRAME_POLL_S)

    def get_frame(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SensorGUI:
    def __init__(self, root, base_url):
        self.root = root
        self.base_url = base_url
        root.title(f"Sensor Reader — {base_url}")
        root.configure(bg=BG)

        self.selected_node = None      # None until readings tell us what nodes exist
        self.node_ids = []
        self.cards = {}                # (node_id, label) -> card widget dict
        self._imgtk = None
        self._last_frame_shape = None  # (h, w) of the last decoded raw frame, for box scaling
        self._clock_var = tk.StringVar(value=time.strftime("%H:%M:%S"))

        self.readings_poller = ReadingsPoller(base_url)
        self.readings_poller.start()
        self.frame_poller = FramePoller(base_url, lambda: self.selected_node)
        self.frame_poller.start()

        self._build_layout()
        self._render_loop()
        self._poll_readings()
        self._refresh_card_states()

    # ---- layout ----
    def _build_layout(self):
        self.root.minsize(1180, 690)
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=18, pady=18)
        main.grid_columnconfigure(0, weight=0)
        main.grid_columnconfigure(1, weight=1, minsize=360)
        main.grid_rowconfigure(0, weight=1)

        # left: video canvas
        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="n")

        video_panel = tk.Frame(left, bg=SURFACE, highlightbackground="#2d3441",
                               highlightthickness=1)
        video_panel.pack(fill="x")

        video_head = tk.Frame(video_panel, bg=SURFACE)
        video_head.pack(fill="x", padx=14, pady=(12, 10))
        tk.Label(video_head, text="Live View", bg=SURFACE, fg=TEXT,
                 font=FONT_TITLE, anchor="w").pack(side="left")
        tk.Label(video_head, textvariable=self._clock_var, bg=PANEL_ALT, fg=MUTED,
                 font=FONT_SMALL, padx=10, pady=4).pack(side="right")

        node_bar = tk.Frame(video_panel, bg=SURFACE)
        node_bar.pack(fill="x", padx=14, pady=(0, 10))
        tk.Label(node_bar, text="Node:", bg=SURFACE, fg=MUTED, font=FONT).pack(side="left")
        self.node_var = tk.StringVar(value="")
        self.node_menu = ttk.Combobox(node_bar, textvariable=self.node_var, state="readonly",
                                      values=[], width=24)
        self.node_menu.pack(side="left", padx=(8, 0))
        self.node_menu.bind("<<ComboboxSelected>>", self._on_node_selected)

        canvas_wrap = tk.Frame(video_panel, bg="#05070a", highlightbackground="#0b0d12",
                               highlightthickness=1)
        canvas_wrap.pack(padx=14, pady=(0, 12))
        self.canvas = tk.Canvas(canvas_wrap, width=DISP_W, height=DISP_H,
                                bg="black", highlightthickness=0)
        self.canvas.pack()

        hint = tk.Label(
            video_panel, bg=SURFACE, fg=MUTED, anchor="w", font=FONT,
            text="Boxes shown are read-only, defined by the Pi's config/nodes.json.",
        )
        hint.pack(fill="x", padx=14, pady=(0, 12))

        btnbar = tk.Frame(left, bg=BG)
        btnbar.pack(fill="x", pady=(14, 10))
        self._mk_button(btnbar, "Refresh Now", self._refresh_now, BLUE).pack(side="left")
        self.auto_btn = self._mk_button(btnbar, "Auto Refresh: ON", self._toggle_auto, ACCENT)
        self.auto_btn.pack(side="left", padx=8)
        self.auto_refresh = True

        self.status_var = tk.StringVar(value="starting...")
        status_bar = tk.Frame(left, bg=SURFACE, highlightbackground="#2d3441", highlightthickness=1)
        status_bar.pack(fill="x", pady=(2, 0))
        tk.Label(status_bar, text="Status", bg=SURFACE, fg=MUTED, font=FONT_BOLD,
                 anchor="w").pack(side="left", padx=(12, 8), pady=8)
        tk.Label(status_bar, textvariable=self.status_var, bg=SURFACE, fg=TEXT, font=FONT,
                 anchor="w").pack(side="left", fill="x", expand=True, padx=(0, 12), pady=8)

        # right: readings panel
        right = tk.Frame(main, bg=PANEL, highlightbackground="#303846", highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        dashboard_head = tk.Frame(right, bg=PANEL)
        dashboard_head.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 12))
        dashboard_head.grid_columnconfigure(0, weight=1)
        tk.Label(dashboard_head, text="Readings", bg=PANEL, fg=TEXT,
                 font=FONT_TITLE, anchor="w").grid(row=0, column=0, sticky="w")
        self.count_var = tk.StringVar(value="0 channels")
        tk.Label(dashboard_head, textvariable=self.count_var, bg=PANEL_ALT, fg=MUTED,
                 font=FONT_SMALL, padx=10, pady=4).grid(row=0, column=1, sticky="e")
        tk.Label(dashboard_head, text="Values pulled from the Pi's pi_server.py",
                 bg=PANEL, fg=MUTED, font=FONT, anchor="w").grid(
                     row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        scroll_area = tk.Frame(right, bg=PANEL)
        scroll_area.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        scroll_area.grid_rowconfigure(0, weight=1)
        scroll_area.grid_columnconfigure(0, weight=1)

        self.cards_canvas = tk.Canvas(scroll_area, bg=PANEL, highlightthickness=0)
        self.cards_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(scroll_area, orient="vertical", command=self.cards_canvas.yview,
                                 bg=PANEL, troughcolor=PANEL, activebackground=PANEL_ALT,
                                 highlightthickness=0, relief="flat")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.cards_canvas.configure(yscrollcommand=scrollbar.set)

        self.cards_frame = tk.Frame(self.cards_canvas, bg=PANEL)
        self.cards_window = self.cards_canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        self.cards_frame.bind("<Configure>", self._on_cards_configure)
        self.cards_canvas.bind("<Configure>", self._on_cards_canvas_configure)

        self.empty_state = tk.Label(
            self.cards_frame,
            text="Waiting for readings from the Pi...",
            bg=PANEL, fg=MUTED, font=FONT, anchor="center", pady=40,
        )
        self.empty_state.pack(fill="x", padx=12, pady=24)

    def _mk_button(self, parent, text, cmd, color):
        return tk.Button(
            parent, text=text, command=cmd, bg=color, fg="#0f1117",
            activebackground=color, activeforeground="#0f1117",
            relief="flat", bd=0, padx=14, pady=8,
            font=FONT_BOLD, cursor="hand2",
        )

    def _on_cards_configure(self, _event=None):
        self.cards_canvas.configure(scrollregion=self.cards_canvas.bbox("all"))

    def _on_cards_canvas_configure(self, event):
        self.cards_canvas.itemconfigure(self.cards_window, width=event.width)

    def _on_node_selected(self, _event=None):
        self.selected_node = self.node_var.get() or None

    def _toggle_auto(self):
        self.auto_refresh = not self.auto_refresh
        self.auto_btn.config(text=f"Auto Refresh: {'ON' if self.auto_refresh else 'OFF'}")

    def _refresh_now(self):
        # background pollers already refresh continuously; this just forces
        # an immediate re-render from whatever they most recently fetched.
        self._apply_readings(self.readings_poller.get())

    # ---- readings panel ----
    def _poll_readings(self):
        if self.auto_refresh:
            data = self.readings_poller.get()
            self._apply_readings(data)
        self.status_var.set(f"readings: {self.readings_poller.status} | frame: {self.frame_poller.status}")
        self.root.after(300, self._poll_readings)

    def _apply_readings(self, data):
        nodes = data.get("nodes", {})
        node_ids = sorted(nodes.keys())
        if node_ids and node_ids != self.node_ids:
            self.node_ids = node_ids
            self.node_menu["values"] = node_ids
            if not self.selected_node or self.selected_node not in node_ids:
                self.selected_node = node_ids[0]
                self.node_var.set(self.selected_node)

        now = time.time()
        seen = set()
        for node_id, node_data in nodes.items():
            for label, box in node_data.get("boxes", {}).items():
                key = (node_id, label)
                seen.add(key)
                if key not in self.cards:
                    self._add_card(key)
                self._update_card(key, box, now)

        for key in list(self.cards.keys()):
            if key not in seen:
                self._remove_card(key)

        self._update_count()
        self._paint_card_states()

    def _add_card(self, key):
        node_id, label = key
        if self.cards == {}:
            self.empty_state.pack_forget()

        outer = tk.Frame(self.cards_frame, bg=CARD_BORDER, padx=1, pady=1)
        outer.pack(fill="x", padx=6, pady=7)
        card = tk.Frame(outer, bg=CARD)
        card.pack(fill="both", expand=True)
        card.grid_columnconfigure(0, weight=1)

        toprow = tk.Frame(card, bg=CARD)
        toprow.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 4))
        toprow.grid_columnconfigure(0, weight=1)
        name_lbl = tk.Label(toprow, text=f"{node_id} / {label}", bg=CARD, fg=TEXT_DARK,
                            font=FONT_BOLD, anchor="w")
        name_lbl.grid(row=0, column=0, sticky="w")

        valrow = tk.Frame(card, bg=CARD)
        valrow.grid(row=1, column=0, sticky="ew", padx=14, pady=(6, 2))
        valrow.grid_columnconfigure(0, weight=1)
        reading_var = tk.StringVar(value="--")
        unit_var = tk.StringVar(value="")
        value_lbl = tk.Label(valrow, textvariable=reading_var, bg=CARD, fg=ACCENT_DIM,
                             font=FONT_VALUE, anchor="w")
        value_lbl.grid(row=0, column=0, sticky="w")
        unit_lbl = tk.Label(valrow, textvariable=unit_var, bg=CARD, fg=TEXT_SOFT,
                            font=FONT_SECTION, anchor="w")
        unit_lbl.grid(row=0, column=1, sticky="sw", padx=(8, 0), pady=(0, 7))

        raw_var = tk.StringVar(value="")
        raw_lbl = tk.Label(card, textvariable=raw_var, bg=CARD, fg=MUTED_DARK,
                           font=FONT_SMALL, anchor="w")
        raw_lbl.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 2))

        foot = tk.Frame(card, bg=CARD)
        foot.grid(row=3, column=0, sticky="ew", padx=14, pady=(4, 12))
        foot.grid_columnconfigure(1, weight=1)
        fresh_dot = tk.Label(foot, text=" ", bg=ACCENT_DIM, width=1, height=1, font=("Helvetica", 4))
        fresh_dot.grid(row=0, column=0, sticky="w")
        freshness_var = tk.StringVar(value="stale")
        freshness_lbl = tk.Label(foot, textvariable=freshness_var, bg=CARD, fg=MUTED_DARK,
                                 font=FONT_SMALL, anchor="w")
        freshness_lbl.grid(row=0, column=1, sticky="w", padx=(5, 10))
        updated_var = tk.StringVar(value="never")
        updated_lbl = tk.Label(foot, textvariable=updated_var, bg=CARD, fg=MUTED_DARK,
                               font=FONT_SMALL, anchor="e")
        updated_lbl.grid(row=0, column=2, sticky="e")

        self.cards[key] = {
            "outer": outer, "card": card, "reading_var": reading_var, "unit_var": unit_var,
            "raw_var": raw_var, "freshness_var": freshness_var, "updated_var": updated_var,
            "value_lbl": value_lbl, "unit_lbl": unit_lbl, "fresh_dot": fresh_dot,
            "freshness_lbl": freshness_lbl, "updated_lbl": updated_lbl, "name_lbl": name_lbl,
            "raw_lbl": raw_lbl,
            "bg_widgets": (card, toprow, valrow, foot, name_lbl, value_lbl, unit_lbl,
                           raw_lbl, freshness_lbl, updated_lbl),
            "last_ts": None, "valid": False,
        }

    def _remove_card(self, key):
        self.cards[key]["outer"].destroy()
        del self.cards[key]
        if not self.cards:
            self.empty_state.pack(fill="x", padx=12, pady=24)

    def _update_card(self, key, box, now):
        c = self.cards[key]
        value = box.get("value")
        valid = bool(box.get("valid"))
        c["reading_var"].set(str(value) if value is not None else "--")
        c["unit_var"].set(box.get("unit") or "")
        raw = box.get("raw_text")
        conf = box.get("confidence")
        c["raw_var"].set(f"raw: {raw!r}  conf={conf}" if raw is not None else "")
        c["last_ts"] = box.get("ts")
        c["valid"] = valid

    def _update_count(self):
        count = len(self.cards)
        label = "channel" if count == 1 else "channels"
        self.count_var.set(f"{count} {label}")

    def _refresh_card_states(self):
        self._clock_var.set(time.strftime("%H:%M:%S"))
        self._paint_card_states()
        self.root.after(1000, self._refresh_card_states)

    def _paint_card_states(self):
        now = time.time()
        for c in self.cards.values():
            ts = c["last_ts"]
            fresh = ts is not None and now - ts <= STALE_AFTER_SECONDS
            valid = c["valid"]
            if fresh and valid:
                border, card_bg, value_fg, dot_bg = CARD_BORDER_FRESH, CARD, "#147a3d", ACCENT
                c["freshness_var"].set("fresh")
            elif fresh and not valid:
                border, card_bg, value_fg, dot_bg = CARD_BORDER_INVALID, CARD, FLAG, FLAG
                c["freshness_var"].set("fresh (invalid)")
            else:
                border, card_bg, value_fg, dot_bg = CARD_BORDER, CARD_STALE, ACCENT_DIM, ACCENT_DIM
                c["freshness_var"].set("stale")

            c["updated_var"].set(time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "never")
            c["outer"].configure(bg=border)
            for widget in c["bg_widgets"]:
                widget.configure(bg=card_bg)
            c["value_lbl"].configure(bg=card_bg, fg=value_fg)
            c["unit_lbl"].configure(bg=card_bg)

    # ---- video render loop ----
    def _render_loop(self):
        frame = self.frame_poller.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            self._last_frame_shape = (h, w)
            disp = cv2.resize(frame, (DISP_W, DISP_H))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self._imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self._imgtk)

            sx, sy = DISP_W / w, DISP_H / h
            node_data = self.readings_poller.get().get("nodes", {}).get(self.selected_node, {})
            for label, box in node_data.get("boxes", {}).items():
                x1, y1 = box["x"] * sx, box["y"] * sy
                x2, y2 = (box["x"] + box["w"]) * sx, (box["y"] + box["h"]) * sy
                color = ACCENT if box.get("valid") else FLAG
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                self.canvas.create_text(x1 + 2, y1 - 8, anchor="w", text=label,
                                        fill=color, font=("Helvetica", 10, "bold"))
        self.root.after(66, self._render_loop)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="pi_server host/IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="pi_server port (default: 8765)")
    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    root = tk.Tk()
    app = SensorGUI(root, base_url)

    def on_close():
        app.readings_poller.stop()
        app.frame_poller.stop()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
