from ultralytics import YOLO
 


model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch

 
# Train the model
model.train(
    data="coco128.yaml",
    project="my_graduation_project",  # 修改根目录名
    name="experiment_01"              # 修改实验名
)
# results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
