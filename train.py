from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load pretrained model
    model = YOLO('./yolov8n.pt')

    # 2. Start training
    results = model.train(
        data='nailong.yaml',   # Dataset config file
        epochs=100,            # Number of training epochs
        imgsz=640,             # Input image size
        batch=8,               # Adjust batch size if GPU memory is insufficient
        device='0',            # '0' for first GPU, change to 'cpu' if no GPU available
        workers=4,             # Number of data loading worker processes
        name='nailong_det'     # Directory name for saving training results
    )
