from ultralytics import YOLO

model = YOLO("yolo11n.pt")

train_results = model.train(
    data="sslad.yaml",  
    epochs=100, 
    imgsz=640, 
    device=0,  
)

metrics = model.val()

results = model("food_train/images/train/1001a01.jpg")
results[0].show()

path = model.export(format="onnx") 