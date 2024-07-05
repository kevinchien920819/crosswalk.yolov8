import onnxruntime as ort
import numpy as np
import cv2

# 定义物件名称
class_names = ['Flexible_delineator_post', 'bus', 'car', 'chair', 'crosswalk', 'motocycle', 'pedestrian', 'trunk']

# 定义图像预处理函数
def preprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (640, 640))
    normalized_image = resized_image / 255.0
    transposed_image = np.transpose(normalized_image, (2, 0, 1))
    input_tensor = np.expand_dims(transposed_image, axis=0).astype(np.float32)
    return input_tensor, image

# 定义后处理函数
def postprocess(outputs, image, conf_threshold=0.5, iou_threshold=0.4):
    boxes, scores, class_ids = [], [], []
    # 检查输出形状和内容
    print(f"Output shape: {outputs.shape}")
    print(f"Output content: {outputs}")

    # 展开维度
    outputs = outputs[0]  # 从 (1, 12, 8400) 变为 (12, 8400)
    
    # 对每一个预测结果进行处理
    for detection in outputs.T:  # 转置后变为 (8400, 12)
        print(f"Detection: {detection}")  # 打印每个检测结果
        if detection[4] > conf_threshold:  # confidence score
            box = detection[:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (center_x, center_y, width, height) = box.astype("int")
            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            scores.append(float(detection[4]))
            class_ids.append(int(detection[5]))

    print(f"Boxes: {boxes}")
    print(f"Scores: {scores}")
    print(f"Class IDs: {class_ids}")

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("No objects detected with the given confidence threshold.")

    return image

# 加载 ONNX 模型
onnx_model_path = r"C:\Users\User\Desktop\crosswalk.v27i.yolov8\runs\detect\train6\weights\best.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# 预处理图像
input_tensor, original_image = preprocess(r"C:\Users\User\Desktop\crosswalk.v27i.yolov8\test.jpg")

# 使用 ONNX 模型进行推理
ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
ort_outs = ort_session.run(None, ort_inputs)
outputs = ort_outs[0]

# 打印输出内容以检查
print(outputs)

# 后处理模型输出
result_image = postprocess(outputs, original_image)

# 显示结果图像
cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
