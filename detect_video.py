import cv2
from ultralytics import YOLO

def realtime_detect_and_save(source='D:\\self\\image_video\\nailong\\video\\20260413.mp4', 
                             model_path='runs/detect/nailong_det/weights/best.pt',
                             save_video_path='D:\\self\\image_video\\nailong\\video\\nailong_detection_output.avi'):
    """
    实时检测并保存视频
    Args:
        source (str): 视频源，'0' 为摄像头，也可以是视频文件路径。
        model_path (str): 训练好的模型路径。
        save_video_path (str): 保存的视频文件名（支持 .avi 或 .mp4）。
    """
    # 1. 加载模型
    model = YOLO(model_path)

    # 2. 打开视频源
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)

    # 3. 获取视频源的帧率、宽高，用于设置输出视频的参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果从摄像头获取，fps 可能为 0，手动设置一个常用值
    if fps == 0:
        fps = 30  # 摄像头默认 30 帧

    # 4. 定义视频编码器和输出对象
    # 在 Windows 上，推荐使用 'DIVX' 编码（生成 .avi 文件）或 'mp4v'（生成 .mp4 文件）
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 若想保存为 .mp4，可改为 *'mp4v'
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

    print(f"开始检测并保存视频，保存路径：{save_video_path}")
    print("按 'q' 键退出。")

    # 5. 循环处理每一帧
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("视频流结束或读取失败，正在退出...")
            break

        # 6. 模型推理
        results = model(frame, verbose=False)

        # 7. 绘制检测结果
        annotated_frame = results[0].plot()

        # 8. 显示实时画面
        cv2.imshow("YOLOv8 奶龙实时检测 - 按 Q 退出", annotated_frame)

        # 9. 将绘制后的帧写入视频文件
        out.write(annotated_frame)

        # 10. 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户按下 Q 键，停止检测。")
            break

    # 11. 释放所有资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频已保存至：{save_video_path}")

if __name__ == '__main__':
    # 使用默认摄像头，保存为 output.avi
    realtime_detect_and_save(source='D:\\self\\image_video\\nailong\\video\\20260413.mp4', save_video_path='D:\\self\\image_video\\nailong\\video\\nailong_detection_output.avi')
    
    # 如果想检测本地视频文件并保存结果，可注释上一行，启用下行：
    # realtime_detect_and_save(source='D:/your_video.mp4', save_video_path='result.avi')