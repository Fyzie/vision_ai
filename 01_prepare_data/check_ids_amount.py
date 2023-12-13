import os

door_id = '/m/02dgv'

train_bboxes_filename = os.path.join('.', 'oidv6-train-annotations-bbox.csv')
validation_bboxes_filename = os.path.join('.', 'validation-annotations-bbox.csv')
test_bboxes_filename = os.path.join('.', 'test-annotations-bbox.csv')

filenames = [train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]

num_data = [0]*len(filenames)
set_ = ['train', 'val', 'test']
for j, filename in enumerate(filenames):
    print(filename)
    with open(filename, 'r') as f:
        line = f.readline()
        while len(line) != 0:
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
            if class_name in [door_id]:
                num_data[j] += 1

            line = f.readline()

        f.close()

    print(f'Number of {set_[j]} data\t: {num_data[j]}')