import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = YOLO("best.pt")  # 可根据需要选择其他的模型

st.title("Object Detection")
uploaded_file = st.file_uploader("选择一个图像文件", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = uploaded_file.read()
    nparr = np.frombuffer(img, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model.predict(img_cv, save=True, imgsz=640, conf=0.5)
    for result in results:
        boxes = result.boxes
        class_names = result.names
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls)]
            confidence = conf.item()
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="检测结果", use_column_width=True)
    output_path = "output_with_boxes.jpg"
    cv2.imwrite(output_path, img_cv)
    st.download_button("下载带框图像", output_path)
