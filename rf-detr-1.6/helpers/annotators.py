# THIS SCRIPT PROVIDES FUNCTIONS TO ANNOTATE IMAGES OUT OF THE PRDICTIONS (BBOX, MASK, CLASS LABELS)
import cv2
import torch
import numpy as np

def mask_annotator(image, masks, colors, alpha=0.3):
    overlay = image.copy()

    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask

        mask_np = mask_np.astype(bool)

        color = np.array(colors[i % len(colors)], dtype=np.uint8)

        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[mask_np] = color

        blended = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
        overlay[mask_np] = blended[mask_np]

    return overlay


def box_annotator(image, boxes, colors, thickness=2):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            colors[i % len(colors)],
            thickness,
            cv2.LINE_AA
        )
    return image


def label_annotator(image, boxes, labels, colors):
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, _, _ = map(int, box)
        color = colors[i % len(colors)]

        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        y1 = max(y1, th + 6)

        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            image, label, (x1 + 2, y1 - 4),
            font, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    return image

# ----- ANNOTATE AN IMAGE ------
def annotate_image(image, detections, class_names, colors):
    if detections is None or len(detections.xyxy) == 0:
        return image
    
    boxes = detections.xyxy
    masks = getattr(detections, 'mask', None)
    
    scores = detections.confidence
        
    class_ids = detections.class_id
    detected_class = [class_names[cid] for cid in list(class_ids)]
    
    mapped_colors = [colors[name] for name in detected_class]

    if scores is None:
        labels = [
            f"{c}"
            for c in detected_class
        ]
    else:
        labels = [
            f"{c}: {s:.2f}"
            for c, s in zip(detected_class, scores)
        ]

    image = np.array(image)
    
    if masks is not None:
        image = mask_annotator(image, masks, mapped_colors, alpha=0.1)
    
    image = box_annotator(image, boxes, mapped_colors)
    image = label_annotator(image, boxes, labels, mapped_colors)

    return image