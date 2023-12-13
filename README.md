# Visual AI with YOLOv8
## Open Datasets Collection
Open Images Dataset V7:  [[Visualizer](https://storage.googleapis.com/openimages/web/visualizer/index.html)] [[Download](https://storage.googleapis.com/openimages/web/download_v7.html)]

### Steps to manually download specific datasets:

![Screenshot 2023-12-12 175820](https://github.com/Fyzie/Visual-AI-with-YoloV8/assets/76240694/938aa6f9-0c42-4d2f-ba3e-784174ea44c1)

1. Download [downloader.py](https://raw.githubusercontent.com/openimages/dataset/master/downloader.py) file.
2. Download the object detection dataset: [Train](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv), [Validation](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv) and [Test](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv).
3. Download metadata to identify label name for specific classes/ datasets. Eg. for bbox: [Boxable class names](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)
> Download annotations and metadata files according to your type of computer vision applications   
4. Execute [create_image_list_file.py](https://github.com/Fyzie/Visual-AI-with-YoloV8/blob/main/01_prepare_data/create_image_list_file.py) (Modify desired class id, directory path, and number of desired images accordingly)
> You may check the number of data exists within each dataset for the ids through [check_image_amount.py](https://github.com/Fyzie/Visual-AI-with-YoloV8/blob/main/01_prepare_data/check_image_amount.py)
5. Execute downloader.py at terminal prompt (make sure you are in the right directory path)
```
python downloader.py $IMAGE_LIST_FILE --download_folder=$DOWNLOAD_FOLDER
```
- $IMAGE_LIST_FILE : file name created from create_image_list_file.py (eg. image_list_file)
- $DOWNLOAD_FOLDER : folder name to store the downloaded datasets (any desired name)
6. Execute [create_dataset_yolo_format.py](https://github.com/Fyzie/Visual-AI-with-YoloV8/blob/main/01_prepare_data/create_dataset_yolo_format.py) (Modify your base directory, class id)
> YOLO label format: class x_center y_center width height   

> Reference: [computervisioneng](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide)

To visualize bbox labels, you may use [visualize_bbox.py](https://github.com/Fyzie/Visual-AI-with-YoloV8/blob/main/01_prepare_data/visualize_bbox.py)

## Object Detection
### Training Using Custom Datasets (Open Images)
1. Create {any_name}.yaml file
```
path: .../data # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/validation  # val images (relative to 'path')

# Classes
names:
  0: door
```
2. Execute [scratch_object_detection.py](https://github.com/Fyzie/Visual-AI-with-YoloV8/blob/main/02_object_detection/scratch_object_detection.py)
3. Once training is completed, get your results

![Screenshot 2023-12-12 210547](https://github.com/Fyzie/Visual-AI-with-YoloV8/assets/76240694/e273b713-cf2d-406b-a27d-cca5126914f4)



