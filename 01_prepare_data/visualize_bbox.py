import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def visualize_boxes(image_path, boxes):
    # Open the image
    image = Image.open(image_path)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Add bounding boxes to the image
    for box in boxes:
        class_label, x, y, width, height = box
        x, y, width, height = float(x), float(y), float(width), float(height)

        # Convert relative coordinates to absolute coordinates
        width *= image.width
        height *= image.height

        x *= image.width
        y *= image.height
        x -= (width/2) 
        y -= (height/2)

        print([x, y, width, height])

        # Create a rectangle patch
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none', label=f'Class {class_label}')

        # Add the rectangle to the Axes
        ax.add_patch(rect)

    # Show the plot
    plt.show()

def read_boxes_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    boxes = []
    for line in lines:
        values = line.split()
        class_label = int(values[0])
        coordinates = [float(value) for value in values[1:]]
        boxes.append([class_label] + coordinates)

    return boxes

if __name__ == "__main__":
    base_directory = r'D:\Pycharm Projects\YPPB Projects\yolov8\data'
    image_name = '0030b2710519aced'
    # Replace 'path/to/image.jpg' with the path to your image file
    image_path = os.path.join(base_directory, 'images', 'train', f'{image_name}.jpg')

    # Replace 'path/to/coordinates.txt' with the path to your txt file
    coordinates_path = os.path.join(base_directory, 'labels', 'train', f'{image_name}.txt')

    # Read bounding boxes from the txt file
    bounding_boxes = read_boxes_from_file(coordinates_path)
    print(bounding_boxes)

    # Visualize the image with bounding boxes
    visualize_boxes(image_path, bounding_boxes)
