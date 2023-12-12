from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

model.train(data=".../config.yaml", epochs=3) # get your .yaml file and declare num of epochs
