import os
import json
import cv2
import random
import colorsys
import numpy as np
from tqdm import tqdm

def mask_annotator(image, masks, colors=[(255, 0, 0)], alpha=0.3):
    """
    Overlays masks on image using the specified colors and transparency.

    Args:
        image (np.ndarray): The input RGB image of shape (H, W, 3), dtype uint8.
        masks (List[torch.Tensor]): List of binary masks, each of shape (H, W) and dtype torch.uint8 or torch.bool.
        colors (List[Tuple[int, int, int]]): List of RGB color tuples. Default is red.
        alpha (float, optional): Transparency value for mask overlay. Default is 0.3.

    Returns:
        np.ndarray: Annotated image with masks overlaid.
    """

    overlay = image.copy()

    for i, mask in enumerate(masks):
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy().astype("uint8")
        else:
            mask_np = mask.astype("uint8")
        color   = np.array(colors[i % len(colors)], dtype=np.uint8)

        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_np == 1,
                (1 - alpha) * overlay[:, :, c] + alpha * color[c],
                overlay[:, :, c]
            )

    return overlay.astype(np.uint8)

def box_annotator(image, boxes, colors=[(255, 0, 0)], thickness=2):
    """
    Draws bounding boxes on image.

    Args:
        image (np.ndarray): The input RGB image of shape (H, W, 3), dtype uint8.
        boxes (Union[np.ndarray, List[Tuple[float, float, float, float]]]): Bounding boxes in (x_min, y_min, x_max, y_max) format.
        colors (List[Tuple[int, int, int]]): List of RGB color tuples. Default is red.
        thickness (int, optional): Thickness of box lines. Default is 2.

    Returns:
        np.ndarray: Annotated image with bounding boxes.
    """

    overlay = image.copy()
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max  = map(int, box)
        color                       = colors[i % len(colors)]
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, thickness)

    return overlay

def label_annotator(image, boxes, labels, colors=[(255, 0, 0)], font_scale=0.5, font_thickness=1):
    """
    Draws text labels above bounding boxes on image.

    Args:
        image (np.ndarray): The input RGB image of shape (H, W, 3), dtype uint8.
        boxes (Union[np.ndarray, List[Tuple[float, float, float, float]]]): Bounding boxes in (x_min, y_min, x_max, y_max) format.
        labels (List[str]): List of label strings (e.g. class names or 'class: score').
        colors (List[Tuple[int, int, int]]): List of RGB color tuples. Default is red.
        font_scale (float, optional): Font size scale. Default is 0.5.
        font_thickness (int, optional): Thickness of the text font. Default is 1.

    Returns:
        np.ndarray: Annotated image with labels drawn.
    """

    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x_min, y_min, _, _  = map(int, box)
        color               = colors[i % len(colors)]
        text_size, _        = cv2.getTextSize(label, font, font_scale, font_thickness)

        text_w, text_h      = text_size
        bg_top_left         = (x_min, y_min - text_h - 4)
        bg_bottom_right     = (x_min + text_w + 4, y_min)

        cv2.rectangle(overlay, bg_top_left, bg_bottom_right, color, thickness=-1)

        cv2.putText(overlay, label, (x_min + 2, y_min - 4), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

    return overlay

def generate_colors(num_classes, seed=42):
    colors = []
    random.seed(seed)
    for i in range(num_classes):
        hue         = i / num_classes
        lightness   = 0.7
        saturation  = 0.5
        rgb         = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple(int(255 * x) for x in rgb))
    return colors

def load_coco_annotation(annotation_path):
    with open(annotation_path) as f:
        coco = json.load(f)
    return coco

def group_annotations_by_image(annotations):
    grouped = {}
    for ann in annotations:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped

def visualize_annotation(image, anns, categories, poly_color=(0, 0, 255), label_color=(0, 255, 0)):
    boxes   = []
    labels  = []
    masks   = []

    for ann in anns:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(categories.get(ann["category_id"], str(ann["category_id"])))

        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for seg in ann["segmentation"]:
                if len(seg) >= 6:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

    colors    = [label_color.get(label, (255, 255, 255)) for label in labels]

    annotated = mask_annotator(image, masks, colors=colors, alpha=0.8) if masks else image
    annotated = box_annotator(annotated, boxes, colors=colors, thickness=4)
    annotated = label_annotator(annotated, boxes, labels, font_scale=1, font_thickness=2, colors=colors)
    return annotated

def main(
    image_dir           = None,
    annotation_filename = "_annotations.coco.json"
):

    annotation_path = os.path.join(image_dir, annotation_filename)

    if not os.path.exists(annotation_path):
        print(f"[ERROR] Annotation file not found: {annotation_path}")
        return

    coco = load_coco_annotation(annotation_path)
    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    annotations_by_image = group_annotations_by_image(coco["annotations"])

    class_colors    = generate_colors(len(categories))
    class_colors = {name: color for name, color in zip(categories.values(), class_colors)}

    for img_id, img_info in tqdm(images.items(), desc="Visualizing"):
        # if img_info["file_name"] == "image_20250910_144229_png.rf.a6212617b28b120a45a5d24c90920145.jpg":
        img_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            print(f"[WARNING] Image file not found: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Could not read image: {img_path}")
            continue

        anns = annotations_by_image.get(img_id, [])
        annotated = visualize_annotation(image, anns, categories, label_color=class_colors)

        cv2.namedWindow(img_info["file_name"], cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(img_info["file_name"], (int(annotated.shape[1]*0.3), int(annotated.shape[0]*0.3)))
        cv2.imshow(img_info["file_name"], annotated)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to break
            break
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(
        image_dir           = r"C:\Users\Hafizi\Documents\computer vision\03-lens\data\lens8\train",
        annotation_filename = "_annotations.coco.json"
    )
