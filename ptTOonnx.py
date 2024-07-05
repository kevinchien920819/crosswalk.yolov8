import torch
import torch.onnx
from ultralytics import YOLO

def torch2Onnx():
    # 加载训练好的 PyTorch 模型
    model_path = r'C:\Users\User\Desktop\crosswalk.v27i.yolov8\runs\detect\train7\weights\best.pt'
    model_pytorch = YOLO(model_path)
    
    # 设置模型为评估模式
    model_pytorch.eval()

    # 创建输入张量
    dummy_input = torch.randn(1, 3, 640, 640)

    # 导出模型到 ONNX 格式
    onnx_model_path = 'model.onnx'
    torch.onnx.export(model_pytorch, dummy_input, onnx_model_path, 
                      input_names=['input'], output_names=['output'], 
                      opset_version=11)
    print(f'Model has been converted to ONNX and saved at {onnx_model_path}')

if __name__ == '__main__':
    torch2Onnx()
