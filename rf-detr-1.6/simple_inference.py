### LOAD MODEL ##############################################################################################
from rfdetr import RFDETRSegNano
MODEL_PATH = ""
CLASS_NAMES = {0: "cable", 1: "overlap"}
CONFIDENCE_THRESHOLD = 0.5
model = RFDETRSegNano(pretrain_weights=MODEL_PATH, num_classes=len(CLASS_NAMES))
model.optimize_for_inference()

### MODEL PREDICTIONS ############################################################################
img_result = image.copy()
predictions = model.predict(image, conf_threshold=CONFIDENCE_THRESHOLD)
for box, score, mask, class_id in zip(predictions['xyxy'], predictions['confidence'], predictions['mask'], predictions['class_id']):
    print(mask)
    x1, y1, x2, y2 = map(int, box)

    label = CLASS_NAMES.get(int(class_id), f"Class {class_id}")
    caption = f"{label}: {score:.2f}"

    cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(img_result, caption, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

######################################################################################################

cv2.imshow("Live Video", image)
