from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
# model = YOLO(".../train/weights/best.pt")  # load a pretrained model 

model.train(data=".../config.yaml", project='/saved_dir', epochs=3) # get your .yaml file, set folder to save model, and declare num of epochs

# more details on model.train parameters: https://github.com/ultralytics/ultralytics/issues/3527#issuecomment-1629368462
