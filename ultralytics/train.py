from ultralytics import YOLO
 


model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch

 
# Train the model
results = model.train(data="coco128.yaml", epochs=100, imgsz=640)


#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 