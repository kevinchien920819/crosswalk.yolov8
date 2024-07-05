from ultralytics import YOLO
import cv2
import os

# 加载训练好的模型
model = YOLO(r'C:\Users\User\Desktop\crosswalk.v27i.yolov8\runs\detect\train\weights\best.pt')

# 设置图片路径和输出文件夹
image_path = r'C:\Users\User\Desktop\crosswalk.v27i.yolov8\test.jpg'
output_image_path = r'C:\Users\User\Desktop\crosswalk.v27i.yolov8\label_image\labeled_test.jpg'
output_label_path = r'C:\Users\User\Desktop\crosswalk.v27i.yolov8\label_image\labels\test.txt'

# 创建输出文件夹
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

# 使用模型进行推论
results = model.predict(source=image_path)

# 保存标记文件
with open(output_label_path, 'w') as f:
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0]  # 使用 YOLO 格式
            f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

# 保存图片
result_image = results[0].plot()  # 绘制结果图像
cv2.imwrite(output_image_path, result_image)
print(f'Processed and saved {image_path} and its label')
