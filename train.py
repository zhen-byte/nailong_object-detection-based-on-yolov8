from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载预训练模型 (推荐yolov8n.pt)
    model = YOLO('./yolov8n.pt')

    # 2. 开始训练
    results = model.train(
        data='nailong.yaml', # 你的数据配置文件
        epochs=100,          # 训练轮数
        imgsz=640,           # 输入图片尺寸
        batch=8,             # 根据显存调整，如果显存不足报错，就调小它
        device='0',          # '0'表示第一块GPU，没有GPU则改为 'cpu'
        workers=4,           # 数据加载进程数
        name='nailong_det'   # 训练结果保存的文件夹名
    )

