#!/usr/bin/env python3
"""
standalone_roi_gui.py — ONE-MACHINE ROI drawing + OCR reading tool.

Everything runs in this single process on whichever machine you launch it
on: it opens ITS OWN camera (or a local image file) and runs the OCR engine
locally. There is no network involved and no second machine required. This
is the file to use if you want to run the whole tool on your laptop with a
laptop-attached webcam, OR run it directly on the Pi with a Pi-attached
camera+display -- either way, camera and OCR live in the same place.

If instead you want the camera to live on the Pi while you draw/view on your
laptop, use the OTHER pair of scripts in this folder instead:
    pi_roi_server.py     (run on the Pi)
    laptop_roi_client.py (run on the laptop)
See roi_gui/README.md -- it explains exactly when to use which, step by step.

A pure "draw a box, read what's in it" utility. No VLM, no node/registration
config, no server. It talks directly to the same PaddleOCR-ONNX engine the
main pipeline uses (core/ocr.py + assets/), so what you see here is exactly
what the pipeline would read for that crop.

Source can be:
  - a live camera (cv2.VideoCapture), or
  - a single static image file (jpg/png).

Controls:
  - Open Camera / Load Image...  pick a frame source
  - Snapshot                     (camera mode only) freeze the current frame
                                  so you can draw on something that isn't moving
  - Draw ROI                     toggle draw mode, then click-drag a box on
                                  the canvas and type a label for it
  - click a box (draw mode off)  select it (highlighted), for Delete Selected
  - Delete Selected / Clear All  remove one or all boxes
  - Read Selected / Read All     run OCR on the box crop(s) of the *current*
                                  frame and show raw_text + confidence

This does not write to config/nodes.json and does not touch pipeline.py,
pi_server.py, or anything else in sensor_app -- it's a self-contained sibling
tool for eyeballing ROIs and OCR output while you set things up.

REQUIRES: this machine must have the OCR assets locally
(sensor_app/assets/en_PP-OCRv5_mobile_rec.onnx, ppocrv5_dict.txt,
pi_ocr_reader.py) and a camera attached to IT, not to a remote Pi.

Run (from the sensor_app/ directory, with its venv active):
    python3 roi_gui/standalone_roi_gui.py
See roi_gui/README.md for full setup instructions.
"""
import sys
import time
from pathlib import Path

# Make "core" importable regardless of cwd -- core/ocr.py resolves its own
# ASSETS_DIR from __file__, so this is the only path wiring needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk, filedialog, simpledialog, messagebox
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with: pip install opencv-python numpy pillow\n"
        "If tkinter is missing on Linux: sudo apt install python3-tk"
    )

from core.ocr import get_default_reader, OCR_MODE

DISP_W, DISP_H = 960, 540
CAMERA_POLL_MS = 66  # ~15fps live preview

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

MIN_BOX_PX = 6  # ignore accidental micro-drags, in canvas pixels


class ROIBox:
    __slots__ = ("label", "fx", "fy", "fw", "fh", "raw_text", "confidence", "read_ts")

    def __init__(self, label, fx, fy, fw, fh):
        self.label = label
        # fractions (0..1) of the displayed canvas, so a box drawn on one
        # frame size still lands in the right place after switching source
        self.fx, self.fy, self.fw, self.fh = fx, fy, fw, fh
        self.raw_text = None
        self.confidence = None
        self.read_ts = None

    def canvas_rect(self):
        return (self.fx * DISP_W, self.fy * DISP_H,
                (self.fx + self.fw) * DISP_W, (self.fy + self.fh) * DISP_H)

    def crop(self, frame):
        h, w = frame.shape[:2]
        x0 = max(0, int(round(self.fx * w)))
        y0 = max(0, int(round(self.fy * h)))
        x1 = min(w, int(round((self.fx + self.fw) * w)))
        y1 = min(h, int(round((self.fy + self.fh) * h)))
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1]


class ROIGui:
    def __init__(self, root):
        self.root = root
        root.title("ROI Reader — draw a box, read what's inside it")
        root.configure(bg=BG)
        root.minsize(1180, 690)

        self.reader = get_default_reader()

        self.cap = None                # cv2.VideoCapture when in camera mode
        self.frame = None              # currently displayed BGR frame
        self.live = False              # camera mode is actively polling
        self.draw_mode = False
        self.boxes = []                # list[ROIBox]
        self.selected = None
        self._drag_start = None
        self._drag_rect_id = None
        self._imgtk = None
        self._camera_after_id = None

        self._build_layout()
        self._set_status(f"OCR engine: {OCR_MODE.upper()}  |  no frame loaded yet")

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
        tk.Label(head, text="ROI Reader", bg=SURFACE, fg=TEXT, font=FONT_TITLE, anchor="w").pack(side="left")

        canvas_wrap = tk.Frame(video_panel, bg="#05070a", highlightbackground="#0b0d12", highlightthickness=1)
        canvas_wrap.pack(padx=14, pady=(0, 12))
        self.canvas = tk.Canvas(canvas_wrap, width=DISP_W, height=DISP_H, bg="black", highlightthickness=0,
                                 cursor="tcross")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # source row
        src_bar = tk.Frame(video_panel, bg=SURFACE)
        src_bar.pack(fill="x", padx=14, pady=(0, 8))
        self._mk_button(src_bar, "Open Camera", self._open_camera, BLUE).pack(side="left")
        self._mk_button(src_bar, "Load Image...", self._load_image, BLUE).pack(side="left", padx=8)
        self.snapshot_btn = self._mk_button(src_bar, "Snapshot", self._snapshot, WARN)
        self.snapshot_btn.pack(side="left")
        self.snapshot_btn.config(state="disabled")

        # roi row
        roi_bar = tk.Frame(video_panel, bg=SURFACE)
        roi_bar.pack(fill="x", padx=14, pady=(0, 8))
        self.draw_btn = self._mk_button(roi_bar, "Draw ROI: OFF", self._toggle_draw, PANEL_ALT)
        self.draw_btn.pack(side="left")
        self._mk_button(roi_bar, "Delete Selected", self._delete_selected, FLAG).pack(side="left", padx=8)
        self._mk_button(roi_bar, "Clear All", self._clear_all, FLAG).pack(side="left")

        # read row
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

        # right: readings list
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

    def _set_status(self, text):
        self.status_var.set(text)

    # ---- source: camera ----------------------------------------------
    def _open_camera(self):
        self._stop_camera()
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open camera index 0")
        except Exception as e:
            messagebox.showerror("Camera", str(e))
            return
        self.cap = cap
        self.live = True
        self.snapshot_btn.config(state="normal")
        self._set_status("Camera live -- click Snapshot to freeze a frame for drawing.")
        self._camera_loop()

    def _camera_loop(self):
        if not self.live or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.frame = frame
            self._render_frame()
        self._camera_after_id = self.root.after(CAMERA_POLL_MS, self._camera_loop)

    def _stop_camera(self):
        self.live = False
        if self._camera_after_id is not None:
            self.root.after_cancel(self._camera_after_id)
            self._camera_after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.snapshot_btn.config(state="disabled")

    def _snapshot(self):
        if self.frame is None:
            return
        self.live = False
        if self._camera_after_id is not None:
            self.root.after_cancel(self._camera_after_id)
            self._camera_after_id = None
        self._set_status("Frame frozen. Draw ROIs, then Read Selected/All. Reopen camera to go live again.")

    # ---- source: static image ----------------------------------------
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Load image",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Load Image", f"Could not read image:\n{path}")
            return
        self._stop_camera()
        self.frame = frame
        self._render_frame()
        self._set_status(f"Loaded {Path(path).name} ({frame.shape[1]}x{frame.shape[0]})")

    # ---- rendering ------------------------------------------------------
    def _render_frame(self):
        if self.frame is None:
            return
        disp = cv2.resize(self.frame, (DISP_W, DISP_H))
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

    # ---- drawing ROIs -----------------------------------------------------
    def _toggle_draw(self):
        self.draw_mode = not self.draw_mode
        self.draw_btn.config(text=f"Draw ROI: {'ON' if self.draw_mode else 'OFF'}",
                              bg=ACCENT if self.draw_mode else PANEL_ALT)

    def _on_press(self, event):
        if self.frame is None:
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
            self._render_frame()
            return
        label = label.strip()
        if any(b.label == label for b in self.boxes):
            messagebox.showwarning("New ROI", f"A box named {label!r} already exists.")
            self._render_frame()
            return

        box = ROIBox(label, x0 / DISP_W, y0 / DISP_H, (x1 - x0) / DISP_W, (y1 - y0) / DISP_H)
        self.boxes.append(box)
        self.selected = box
        self._add_card(box)
        self._render_frame()
        self._update_count()

    def _select_at(self, cx, cy):
        hit = None
        for box in reversed(self.boxes):
            x1, y1, x2, y2 = box.canvas_rect()
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                hit = box
                break
        self.selected = hit
        self._render_frame()
        self._refresh_card_selection()

    def _delete_selected(self):
        if self.selected is None:
            return
        self._remove_box(self.selected)
        self.selected = None
        self._render_frame()

    def _clear_all(self):
        if not self.boxes:
            return
        if not messagebox.askyesno("Clear All", f"Remove all {len(self.boxes)} box(es)?"):
            return
        for box in list(self.boxes):
            self._remove_box(box)
        self.selected = None
        self._render_frame()

    def _remove_box(self, box):
        if box in self.boxes:
            self.boxes.remove(box)
        card = self._cards.pop(box, None) if hasattr(self, "_cards") else None
        if card:
            card["outer"].destroy()
        self._update_count()
        if not self.boxes:
            self.empty_state.pack(fill="x", padx=12, pady=24)

    # ---- reading ----------------------------------------------------------
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
        if self.frame is None:
            messagebox.showinfo("Read", "Load a camera frame or image first.")
            return
        now = time.time()
        for box in boxes:
            crop = box.crop(self.frame)
            if crop is None or crop.size == 0:
                box.raw_text, box.confidence = None, None
            else:
                box.raw_text, box.confidence = self.reader.read_text(crop, key=box.label)
            box.read_ts = now
            self._update_card(box)
        self._set_status(f"Read {len(boxes)} box(es) at {time.strftime('%H:%M:%S', time.localtime(now))}")

    # ---- readings panel cards ----------------------------------------------
    def _add_card(self, box):
        if not hasattr(self, "_cards"):
            self._cards = {}
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
        self._render_frame()
        self._refresh_card_selection()

    def _refresh_card_selection(self):
        for b, c in getattr(self, "_cards", {}).items():
            c["outer"].configure(bg=ACCENT if b is self.selected else "#dbe2ec")

    def _update_count(self):
        count = len(self.boxes)
        self.count_var.set(f"{count} {'box' if count == 1 else 'boxes'}")

    # ---- lifecycle ----------------------------------------------------------
    def close(self):
        self._stop_camera()


def main():
    root = tk.Tk()
    app = ROIGui(root)

    def on_close():
        app.close()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
