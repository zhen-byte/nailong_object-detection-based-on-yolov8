import cv2
from ultralytics import YOLO

def camera_detect(model_path='runs/detect/nailong_det/weights/best.pt', camera_id=0):
    """
    Open the computer's webcam and run real-time Nailong detection. Press Q to exit.
    Args:
        model_path (str): Path to the trained model weights.
        camera_id (int): Camera device ID, defaults to 0 (built-in webcam).
    """
    # 1. Load the model
    model = YOLO(model_path)

    # 2. Open the camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Cannot open camera, please check if the device is in use.")
        return

    # 3. Get frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Camera opened, starting real-time Nailong detection!")
    print("Press 'q' to exit.")
    print(f"Frame resolution: {width} x {height}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Camera read failed, exiting...")
            break

        # 4. Run model inference
        results = model(frame, verbose=False)

        # 5. Draw detection results
        annotated_frame = results[0].plot()

        # 6. Display the live feed
        cv2.imshow("Nailong Detection - Press Q to Exit", annotated_frame)

        # 7. Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed Q, stopping detection.")
            break
    # 8. Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")