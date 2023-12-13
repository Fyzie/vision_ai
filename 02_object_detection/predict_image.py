import os
from ultralytics import YOLO
import cv2

def detect(image, model_path):
    # Load a model
    model = YOLO(model_path)  # load a custom model

    threshold = 0.5

    results = model(image)[0]
    print(results)

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Object Detection', image)
    cv2.waitKey()


path = r'...\data\images\test\0938634ea64e52a9.jpg'
image = cv2.imread(path)
# cv2.imshow('Original Image', image)
# cv2.waitKey()

model_path = os.path.join('results', 'train', 'weights', 'last.pt')
detect(image, model_path)
