
import cv2
import numpy as np

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

    if len(overlay.shape) == 2 or overlay.shape[2] == 1:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for i, mask in enumerate(masks):
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy().astype("uint8")
        else:
            mask_np = mask.astype("uint8")

        color = np.array(colors[i % len(colors)], dtype=np.uint8)

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
