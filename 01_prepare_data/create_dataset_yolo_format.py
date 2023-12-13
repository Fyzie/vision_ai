# https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/tree/master/prepare_data

import os
import shutil
import time

base_directory = 'D:\Pycharm Projects\YPPB Projects\yolov8'

DATA_ALL_DIR = os.path.join(base_directory, '$DOWNLOAD_FOLDER')

DATA_OUT_DIR = os.path.join(base_directory, 'data_2')

for set_ in ['train', 'validation', 'test']:
    for dir_ in [os.path.join(DATA_OUT_DIR, 'images', set_),
                 os.path.join(DATA_OUT_DIR, 'labels', set_)]:
        if os.path.exists(dir_):
            shutil.rmtree(dir_)
        os.makedirs(dir_)

door_id = '/m/02dgv'

train_bboxes_filename = os.path.join(base_directory, 'oidv6-train-annotations-bbox.csv')
validation_bboxes_filename = os.path.join(base_directory, 'validation-annotations-bbox.csv')
test_bboxes_filename = os.path.join(base_directory, 'test-annotations-bbox.csv')


for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    set_ = ['train', 'validation', 'test'][j]
    print(filename)
    with open(filename, 'r') as f:
        line = f.readline()

        while len(line) != 0:
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
            input_image = os.path.join(DATA_ALL_DIR, '{}.jpg'.format(id))
            if class_name in [door_id] and os.path.exists(input_image):
                output_image = os.path.join(DATA_OUT_DIR, 'images', set_, '{}.jpg'.format(id))
                if not os.path.exists(output_image):
                    shutil.copy(input_image, output_image)
                output_label_file = os.path.join(DATA_OUT_DIR, 'labels', set_, '{}.txt'.format(id))
                with open(output_label_file, 'a') as f_ann:
                    # class_id, xc, yx, w, h
                    x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]

                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    f_ann.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                    f_ann.close()

            line = f.readline()

        
    
