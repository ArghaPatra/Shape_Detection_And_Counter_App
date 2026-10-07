"""Shape Counter: desktop interface for the shape detection and counting system.

Run from the project root:   python app.py
Build an .exe with:          build_exe.bat
"""
import os

# Two copies of the OpenMP runtime (numpy/MKL and others) can abort the program on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import base64
import sys
import traceback
from collections import Counter

import cv2
import joblib
import numpy as np
import sklearn.ensemble  # noqa: F401  (imported so PyInstaller bundles the model's classes)
import sklearn.tree      # noqa: F401

from src.counting import detect_shapes, draw_result

MAX_SIDE = 1000
CLASSES = ["circle", "triangle", "square", "pentagon", "hexagon"]
LOG_FILE = os.path.join(os.path.expanduser("~"), "shape_counter_error.log")


def resource_path(rel):
    """Path to a bundled file, both when run as a script and as a PyInstaller exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


def load_image(path):
    """Read any common image file (also from folders with non-ASCII names) as 8-bit BGR."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("This file could not be read as an image.")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                       # transparent PNG: put it on white
        alpha = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return img


def analyze(img, bundle):
    """Detect and classify all shapes. Returns (annotated image, per-class Counter, total)."""
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    results = detect_shapes(img, bundle["model"], bundle["features"], robust=True)
    return draw_result(img, results), Counter(label for _, label in results), len(results)


class App:
    def __init__(self, root):
        import tkinter as tk
        self.tk = tk
        self.root = root
        root.title("Shape Counter")
        root.geometry("1100x720")
        root.minsize(760, 480)
        self.bundle = None
        self.image = None       # what is currently displayed (BGR)
        self.result = None      # annotated result (BGR) or None
        self.photo = None

        bar = tk.Frame(root, padx=8, pady=8)
        bar.pack(fill="x")
        tk.Button(bar, text="Open image...", width=14, command=self.open_image).pack(side="left")
        self.save_btn = tk.Button(bar, text="Save result...", width=14, state="disabled",
                                  command=self.save_result)
        self.save_btn.pack(side="left", padx=6)
        self.status = tk.Label(bar, text="Open an image to start.", anchor="w")
        self.status.pack(side="left", padx=12)

        body = tk.Frame(root)
        body.pack(fill="both", expand=True)
        side = tk.Frame(body, width=210, padx=14, pady=10)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="Total shapes", font=("Segoe UI", 11)).pack(anchor="w")
        self.total = tk.Label(side, text="-", font=("Segoe UI", 40, "bold"))
        self.total.pack(anchor="w", pady=(0, 14))
        self.rows = {}
        for c in CLASSES:
            row = tk.Frame(side)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=c.capitalize(), font=("Segoe UI", 11), anchor="w").pack(side="left")
            n = tk.Label(row, text="-", font=("Segoe UI", 11, "bold"), anchor="e")
            n.pack(side="right")
            self.rows[c] = n
        tk.Label(side, wraplength=190, justify="left", fg="#666", font=("Segoe UI", 9),
                 text="Only these five shapes are known. Other shapes are assigned to the "
                      "nearest of them.").pack(side="bottom", anchor="w")

        self.canvas = tk.Canvas(body, bg="#f2f2f2", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.show())
        self.canvas.bind("<Double-Button-1>", lambda e: self.open_image())

    # ----- model -----
    def get_bundle(self):
        if self.bundle is None:
            p = resource_path(os.path.join("models", "shape_rf.joblib"))
            if not os.path.exists(p):
                raise FileNotFoundError(f"Model file not found:\n{p}\n\nRun train_model.py first.")
            self.bundle = joblib.load(p)
        return self.bundle

    # ----- actions -----
    def open_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if path:
            self.run(path)

    def run(self, path):
        try:
            self.image = load_image(path)
            self.result = None
            self.save_btn.config(state="disabled")
            self.show()
            self.status.config(text="Analysing...")
            self.root.update()
            out, counts, total = analyze(self.image, self.get_bundle())
            self.result = out
            self.image = out
            self.show()
            self.total.config(text=str(total))
            for c in CLASSES:
                self.rows[c].config(text=str(counts.get(c, 0)))
            self.status.config(text=os.path.basename(path))
            self.save_btn.config(state="normal")
        except Exception as e:
            log_error()
            self.status.config(text="Failed.")
            from tkinter import messagebox
            messagebox.showerror("Shape Counter", str(e))

    def save_result(self):
        from tkinter import filedialog, messagebox
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG image", "*.png")])
        if path:
            try:
                cv2.imencode(".png", self.result)[1].tofile(path)
                self.status.config(text="Saved " + os.path.basename(path))
            except Exception as e:
                messagebox.showerror("Shape Counter", str(e))

    # ----- drawing -----
    def show(self):
        c = self.canvas
        cw, ch = c.winfo_width(), c.winfo_height()
        c.delete("all")
        if self.image is None or cw < 10 or ch < 10:
            c.create_text(cw // 2, ch // 2, text="Open an image (or double-click here)", fill="#888",
                          font=("Segoe UI", 14))
            return
        h, w = self.image.shape[:2]
        s = min(cw / w, ch / h)
        disp = cv2.resize(self.image, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        data = base64.b64encode(cv2.imencode(".png", disp)[1].tobytes()).decode("ascii")
        self.photo = self.tk.PhotoImage(data=data)
        c.create_image(cw // 2, ch // 2, image=self.photo)


def log_error():
    try:
        with open(LOG_FILE, "a") as f:
            f.write(traceback.format_exc() + "\n")
    except OSError:
        pass


def main():
    import tkinter as tk
    try:                                         # sharp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_error()
        raise