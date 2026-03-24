from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from helpers.viz import *

from PIL import Image
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from helpers.viz import *

"""
inference  for SAHI sliced prediction.
"""
class SlicedDetections:
    def __init__(self, boxes, scores, masks, class_ids):
        self.xyxy = np.array(boxes)
        self.confidence = np.array(scores)
        self.mask = masks # List of numpy arrays
        self.class_id = np.array(class_ids)

def predict_sliced_custom(model, img_np, target_slices, overlap_px, conf_thresh=0.2):
    h, w = img_np.shape[:2]
    
    overlap_w, overlap_h = overlap_px[0], overlap_px[1]
    slice_w, slice_h = w // target_slices[0], h // target_slices[1]
    
    slice_coords = [
        [0, 0, slice_w + overlap_w, slice_h + overlap_h],           # Top-Left
        [slice_w - overlap_w, 0, w, slice_h + overlap_h],           # Top-Right
        [0, slice_h - overlap_h, slice_w + overlap_w, h],           # Bottom-Left
        [slice_w - overlap_w, slice_h - overlap_h, w, h]            # Bottom-Right
    ]

    image_list = [img_np[sc[1]:sc[3], sc[0]:sc[2]] for sc in slice_coords]

    batch_results = model.predict(image_list, return_masks=True, threshold=conf_thresh)

    all_boxes, all_masks, all_scores, all_ids = [], [], [], []

    for i, det in enumerate(batch_results):
        ox, oy = slice_coords[i][0], slice_coords[i][1]
        
        b = det.xyxy
        s = det.confidence
        m = det.mask
        cid = det.class_id

        for idx in range(len(b)):
            bx1, by1, bx2, by2 = b[idx]
            all_boxes.append([bx1 + ox, by1 + oy, bx2 + ox, by2 + oy])
            all_scores.append(s[idx])
            all_ids.append(cid[idx])
            
            full_mask = np.zeros((h, w), dtype=bool)
            slice_mask = m[idx].cpu().numpy() if hasattr(m[idx], 'cpu') else m[idx]
            slice_mask = slice_mask.astype(bool)
            
            mh, mw = slice_mask.shape[:2]
            full_mask[oy : oy + mh, ox : ox + mw] = slice_mask
            all_masks.append(full_mask)

    return SlicedDetections(all_boxes, all_scores, all_masks, all_ids)

def run_inference_sliced(model, images, output_dir, class_names, colors, inference_batch, target_slices, overlap_px, sample_size=16):

    annotated = []
    sampled_paths = random.sample(images, min(sample_size, len(images)))
    model.optimize_for_inference(batch_size = inference_batch) # inference batch == total number of slices e.g. (2, 2) = 2*2 = 4

    for img_path in sampled_paths:
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        detections = predict_sliced_custom(model, img_np, target_slices, overlap_px, conf_thresh=0.2)

        res_img = annotate_image(img_np, detections, class_names, colors)
        annotated.append(res_img)

    save_grid(annotated, output_dir)


"""
inference  for standard prediction.
"""
def run_inference(model, images, output_dir, class_names, colors):
    annotated = []
    model.optimize_for_inference() # inference optimization by default on batch size 1
    for img_path in random.sample(images, min(16, len(images))):
        img = Image.open(img_path).convert("RGB")
        detections = model.predict(img, threshold=0.2)
        annotated.append(
            annotate_image(img, detections, class_names, colors)
        )

    save_grid(annotated, output_dir)

def save_grid(annotated, output_dir):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, img in zip(axes.flat, annotated):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "annotated_grid.png"), dpi=200)
    plt.show()