# Visual AI with YOLOv8
## Open Datasets Collection
Open Images Dataset V7:  [[Visualizer](https://storage.googleapis.com/openimages/web/visualizer/index.html)] [[Download](https://storage.googleapis.com/openimages/web/download_v7.html)]

### Steps to manually download specific datasets:
![image](https://github.com/Fyzie/Visual-AI-with-YoloV8/assets/76240694/2878b7a9-fdd6-4e95-b313-5e852655d2a9)   

![Screenshot 2023-12-12 175820](https://github.com/Fyzie/Visual-AI-with-YoloV8/assets/76240694/938aa6f9-0c42-4d2f-ba3e-784174ea44c1)

1. Download [downloader.py](https://raw.githubusercontent.com/openimages/dataset/master/downloader.py) file.
2. Download the object detection dataset: [Train](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv), [Validation](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv) and [Test](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv).
3. Download metadata to identify label name for specific classes/ datasets. Eg. for bbox: [Boxable class names](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)
4. Execute create_image_list_file.py (Modify desired class id, directory path, and number of desired images accordingly)
5. Execute downloader.py at terminal prompt (make sure you are in the right directory path)
```
python downloader.py $IMAGE_LIST_FILE --download_folder=$DOWNLOAD_FOLDER
```
- $IMAGE_LIST_FILE : file name created from create_image_list_file.py (eg. image_list_file)
- $DOWNLOAD_FOLDER : folder name to store the downloaded datasets (any desired name)
6. Execute create_dataset_yolo_format.py, changing DATA_ALL_DIR by $DOWNLOAD_FOLDER

> Download annotations and metadata files according to your type of computer vision applications
> Recommended to keep all scripts in one folder !!!   
> Reference: [computervisioneng](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide)

## Object Detection
