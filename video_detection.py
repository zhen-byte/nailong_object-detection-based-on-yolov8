import cv2
from ultralytics import YOLO

def realtime_detect_and_save(source='D:\\self\\image_video\\nailong\\video\\20260413.mp4',
                             model_path='runs/detect/nailong_det/weights/best.pt',
                             save_video_path='D:\\self\\image_video\\nailong\\video\\nailong_detection_output.avi'):
    """
    Run real-time detection on a video source and save the output.
    Args:
        source (str): Video source — '0' for webcam, or a video file path.
        model_path (str): Path to the trained model weights.
        save_video_path (str): Path to save the output video (supports .avi or .mp4).
    """
    # 1. Load the model
    model = YOLO(model_path)

    # 2. Open the video source
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)

    # 3. Get frame rate and dimensions from the source for output video settings
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # If reading from webcam, FPS may be 0; fall back to a common value
    if fps == 0:
        fps = 30  # Default webcam frame rate

    # 4. Define video codec and output writer
    # On Windows, use 'DIVX' codec (produces .avi) or 'mp4v' (produces .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Change to *'mp4v' for .mp4 output
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

    print(f"Detection started, saving output to: {save_video_path}")
    print("Press 'q' to exit.")

    # 5. Process each frame in a loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream ended or read failed, exiting...")
            break

        # 6. Run model inference
        results = model(frame, verbose=False)

        # 7. Draw detection results
        annotated_frame = results[0].plot()

        # 8. Display the live feed
        cv2.imshow("YOLOv8 Nailong Detection - Press Q to Exit", annotated_frame)

        # 9. Write the annotated frame to the output video
        out.write(annotated_frame)

        # 10. Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed Q, stopping detection.")
            break

    # 11. Release all resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {save_video_path}")