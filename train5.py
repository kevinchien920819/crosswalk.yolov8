from ultralytics import YOLO
import os

if __name__ == '__main__':
    print("Current Working Directory:", os.getcwd())
    data_path = r"C:\Users\User\Desktop\crosswalk.v27i.yolov8\data.yaml"
    print("Using data.yaml from:", data_path)

    if os.path.exists(data_path):
        print("Found data.yaml")
        # 載入模型
        model = YOLO('yolov8n.yaml')  # build a new model from YAML
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

        # 開始訓練，使用完整路徑
        model.train(data=data_path,
                    task="detect",
                    mode="train",
                    imgsz=640,
                    epochs=100,  # 增加訓練輪數
                    batch=64,  # 增加批次大小
                    optimizer="Adam",  # 使用 Adam 優化器
                    lr0=0.001,  # 調整初始學習率
                    lrf=0.00001,  # 調整學習率終值
                    weight_decay=5e-4,  # 添加權重衰減以防止過擬合
                    patience=20,  # 設置提前停止
                    augment=True,  # 啟用數據增強
                    dropout=0.3,  # 加入 Dropout 正則化
                    plots=True,
                    device="0")

        # 訓練完成後導出模型
        print("Training completed. Exporting the model...")

        print("Model exported successfully.")
    else:
        print("data.yaml not found, check the path.")
