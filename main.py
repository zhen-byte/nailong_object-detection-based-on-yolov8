import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
from PIL import Image, ImageTk
from camera_detection import camera_detect
from video_detection import realtime_detect_and_save


# ── helpers ──────────────────────────────────────────────────────────────

MODEL_PATH = 'runs/detect/nailong_det/weights/best.pt'


def check_yolo_model():
     #Check that the trained model exists before launching detection
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror(
            "Model Not Found",
            f"Trained model weights not found at:\n{MODEL_PATH}\n\n"
            "Please run train.py first or check the model path."
        )
        return False
    return True


def set_status(msg):
    status_label.config(text=msg)


# ── callbacks ────────────────────────────────────────────────────────────

def on_camera():
    if not check_yolo_model():
        return
    set_status("Camera detection running… (Press Q in the detection window to stop)")
    threading.Thread(target=camera_detect, daemon=True).start()


def on_video():
    if not check_yolo_model():
        return

    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
            ("All files", "*.*"),
        ],
    )
    if not file_path:
        return

    save_path = filedialog.asksaveasfilename(
        title="Save detection result as…",
        defaultextension=".avi",
        filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("All files", "*.*")],
    )
    if not save_path:
        return

    set_status("Processing video… (Press Q in the detection window to stop)")
    threading.Thread(
        target=lambda: realtime_detect_and_save(
            source=file_path, save_video_path=save_path
        ),
        daemon=True,
    ).start()


def on_exit():
    root.destroy()


# ── GUI layout ───────────────────────────────────────────────────────────

BG_PATH = 'background.jpg'
W = 522
H = 362

root = tk.Tk()
root.title("Nailong Detection System")
root.geometry(f"{W}x{H}")
root.resizable(False, False)

# ── background image ──
bg_pil = Image.open(BG_PATH).resize((W, H), Image.LANCZOS)
bg_tk = ImageTk.PhotoImage(bg_pil)

canvas = tk.Canvas(root, width=W, height=H, highlightthickness=0)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_tk, anchor="nw")
canvas.bg_tk = bg_tk  # keep reference

# ── header (on canvas) ──
canvas.create_text(
    W // 2, 36,
    text="Nailong Detection",
    font=("Segoe UI", 18, "bold"),
    fill="white",
)

# ── buttons (left side, away from Nailong) ──
btn_frame = tk.Frame(canvas, bg="", padx=10, pady=10)
canvas.create_window(115, 195, window=btn_frame)

btn_camera = tk.Button(
    btn_frame,
    text="Camera Detection",
    font=("Segoe UI", 11),
    width=22,
    height=1,
    bg="#FFF3CD",
    activebackground="#FFE69C",
    command=on_camera,
)
btn_camera.pack(pady=5)

btn_video = tk.Button(
    btn_frame,
    text="Video Detection",
    font=("Segoe UI", 11),
    width=22,
    height=1,
    bg="#FFF3CD",
    activebackground="#FFE69C",
    command=on_video,
)
btn_video.pack(pady=5)

btn_exit = tk.Button(
    btn_frame,
    text="Exit",
    font=("Segoe UI", 11),
    width=22,
    height=1,
    bg="#FFF3CD",
    activebackground="#E6D192",
    command=on_exit,
)
btn_exit.pack(pady=5)

# ── status bar (on canvas) ──
status_label = tk.Label(
    root,
    text="Ready — select a mode above.",
    font=("Segoe UI", 9),
    fg="#333333",
    bg="#FFFFFF",
    padx=8,
    pady=2,
)
canvas.create_window(W // 2, H - 16, window=status_label)

root.mainloop()
