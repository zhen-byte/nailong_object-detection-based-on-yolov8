# Nailong Detection — YOLOv8 Custom Object Detection

A custom object detection project that identifies **Nailong**, a cartoon character, using a fine-tuned YOLOv8 model.

## Project Overview

This project trains a YOLOv8 nano model on a custom dataset of labeled video frames to detect Nailong in images and real-time video streams.

### Files

| File | Purpose |
|------|---------|
| `gui.py` | **Graphical launcher** — the easiest way to run the project (double-click `run.bat`) |
| `main.py` | Terminal-based menu (alternative to gui.py) |
| `camera_detection.py` | Real-time detection using a webcam — detects Nailong from live camera feed |
| `video_detection.py` | Inference on a video file — detects Nailong frame by frame, displays and saves the result |
| `train.py` | Training script — loads YOLOv8n pretrained weights and trains on the custom dataset |
| `nailong.yaml` | Dataset configuration — specifies training/validation image paths, 1 class (`nailong`) |

### Dataset

- **Source**: Extracted video frames with hand-labeled bounding boxes
- **Format**: YOLO label format — `class x_center y_center width height` (normalized)
- **Classes**: 1 (`nailong`)
- **Split**: Training (~150+ frames) / Validation (~20+ frames)
- **Directory**: `datasets/nailong_dataset/`

### Training

```bash
python train.py
```

- **Base model**: `yolov8n.pt`
- **Image size**: 640×640
- **Batch size**: 8
- **Epochs**: 100
- **Device**: GPU
- **Output**: `runs/detect/nailong_det/weights/best.pt`

### Inference

**GUI launcher (recommended):**
```bash
python gui.py
```
Or simply double-click `run.bat`.

**Terminal menu (alternative):**
```bash
python main.py
```

Press **Q** to exit the detection window.

### Requirements

- Python 3.8+
- [ultralytics](https://github.com/ultralytics/ultralytics) >= 8.0.0
- opencv-python >= 4.0.0

**Quick install (recommended — use a virtual environment):**
```bash
python -m venv venv
venv\Scripts\activate
pip install ultralytics opencv-python
```

**Verify the model weights exist** at `runs/detect/nailong_det/weights/best.pt` before running inference.

### Trained Weights

The best model is saved at `runs/detect/nailong_det/weights/best.pt`. You can also download pretrained YOLOv8 weights from the [Ultralytics official repository](https://github.com/ultralytics/assets/releases).

### Demo

A sample detection result is available as `detection_example.mp4`.
