# https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/tree/master/prepare_data

import os
import shutil

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

########## if limiter was used before (make sure the numbers are same) ###########
limit = [300, 90, 30] 

for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    set_ = ['train', 'validation', 'test'][j]
    print(filename)
    with open(filename, 'r') as f:
        line = f.readline()

        ############################### if limiter was used before ######################
        num_id = 0 
        num = 0

        while len(line) != 0:
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
            if class_name in [door_id]:
                if not os.path.exists(os.path.join(DATA_OUT_DIR, 'images', set_, '{}.jpg'.format(id))):
                    shutil.copy(os.path.join(DATA_ALL_DIR, '{}.jpg'.format(id)),
                                os.path.join(DATA_OUT_DIR, 'images', set_, '{}.jpg'.format(id)))
                with open(os.path.join(DATA_OUT_DIR, 'labels', set_, '{}.txt'.format(id)), 'a') as f_ann:
                    x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]

                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1

                    f_ann.write('0 {} {} {} {}\n'.format(x, y, w, h))
                    f_ann.close()

                ####################### if limiter was used before ##############################
                num_id += 1
                if num_id == limit[j]:
                    break
                #################################################################################
            num += 1

            line = f.readline()
