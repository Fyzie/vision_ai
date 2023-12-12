# https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/tree/master/prepare_data
# comment all LIMITER lines if want to download all image ids in the specific datasets

import os

door_id = '/m/02dgv'

train_bboxes_filename = os.path.join('.', 'oidv6-train-annotations-bbox.csv')
validation_bboxes_filename = os.path.join('.', 'validation-annotations-bbox.csv')
test_bboxes_filename = os.path.join('.', 'test-annotations-bbox.csv')

image_list_file_path = os.path.join('.', 'image_list_file')

########## LIMITER: limit number of desired image ids for each dataset [train, validation, test] ###########
limit = [300, 90, 30] 

image_list_file_list = []
for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    print(filename)
    with open(filename, 'r') as f:
        line = f.readline()

        ############################### LIMITER: initiate limit indicator######################
        num_id = 0 
        
        while len(line) != 0:
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
            if class_name in [door_id] and id not in image_list_file_list:
                image_list_file_list.append(id)
                with open(image_list_file_path, 'a') as fw:
                    fw.write('{}/{}\n'.format(['train', 'validation', 'test'][j], id))

                ########### LIMITER: limit number of desired image ids accordingly###############
                num_id += 1
                if num_id == limit[j]:
                    break
                #################################################################################

            line = f.readline()

        f.close()
