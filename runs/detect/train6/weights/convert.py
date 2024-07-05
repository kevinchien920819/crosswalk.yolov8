from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO(r"/Users/icebird/Documents/maccode/python/torchtest/crosswalk.v27i.yolov8/runs/detect/train6/weights/best.pt")
model.export(format = "onnx")  # export the model to onnx format
onnx_model = YOLO('/Users/icebird/Documents/maccode/python/torchtest/crosswalk.v27i.yolov8/runs/detect/train6/weights/best.onnx')  # load the onnx model

result = onnx_model("test.jpg")  # inference the image
result_image = result[0].plot()
# image = model("test.jpg")[0].plot()
plt.title("Original Image of ONNX")
# plt.imshow(image)
plt.imshow(result_image)
plt.axis('off')
plt.show()
# Display the result image
# plt.axis('off')
# plt.show()