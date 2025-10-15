# inferencing fasterrcnn and maskrcnn resnet

import os
import cv2
import matplotlib.pyplot as plt

from torchvision import models as mdl

from utils.models import load_metadata, load_model, pred
from utils.annotators import mask_annotator, box_annotator, label_annotator

#---------------------------------------- ANNOTATOR FUNCTIONS ----------------------------------------#

def main(
        image_path      = None,
        metadata_path   = None,
        model_path      = None,
        conf_thresh     = 0.5,
        with_score      = False,
        is_saved        = False
):
    image   = cv2.imread(image_path)
    image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    metadata = load_metadata(metadata_path)

    model   = load_model(model_path, metadata)
    
    results = pred(image, model, metadata, conf_thresh = conf_thresh, with_score=with_score)

    image   = results[0]
    boxes   = results[2]
    masks   = results[3]
    labels  = results[4]
    colors  = results[5]

    original_resized = cv2.resize(image, image.shape[:2][::-1])

    overlay = mask_annotator(image, masks, colors, alpha=0.5) if masks is not None else image
    overlay = box_annotator(overlay, boxes, colors)
    overlay = label_annotator(overlay, boxes, labels, colors)

    if is_saved:
        image_dir, image_file = os.path.split(image_path)
        image_name, image_ext = os.path.splitext(image_file)
        save_path = os.path.join(image_dir, f"detect_{image_name}{image_ext}")

        save_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_overlay)

        print(f"[INFO] Prediction image saved at: {save_path}")


    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(original_resized)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(
        image_path      = r'',
        metadata_path   = r'',
        model_path      = r'',
        conf_thresh     = 0.5,
        is_saved        = True
    )
