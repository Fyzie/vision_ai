import os
import cv2
import json
import shutil
import colorsys
import numpy as np
from tqdm import tqdm

# ----- SLICING IMAGE IN DATASET---------
def slice_dataset(src_dataset_dir, target_slices=(2, 2), overlap_px=(80, 0), output_name = "_sliced"):
    assert isinstance(target_slices, tuple) and len(target_slices) == 2, "target_slices must be a tuple of (rows, cols)"
    assert all(isinstance(x, int) for x in target_slices), "target_slices elements must be integers"
    assert isinstance(overlap_px, tuple) and len(overlap_px) == 2, "overlap_px must be a tuple of (x_overlap, y_overlap)"
    assert all(isinstance(x, int) for x in overlap_px), "overlap_px elements must be integers"

    ov_w, ov_h = overlap_px
    new_dir = src_dataset_dir + output_name
    
    if os.path.exists(new_dir):
        print(f"Directory {new_dir} already exists. Skip slicing.")
        return new_dir

    for split in ["train", "valid", "test"]:
        split_src = os.path.join(src_dataset_dir, split)
        split_dst = os.path.join(new_dir, split)
        if not os.path.exists(split_src): continue
        
        os.makedirs(split_dst, exist_ok=True)
        ann_path = os.path.join(split_src, "_annotations.coco.json")
        with open(ann_path, 'r') as f:
            coco = json.load(f)

        new_images = []
        new_anns = []
        img_id_count = 1
        ann_id_count = 1

        for img_info in tqdm(coco['images'], desc=f"Slicing {split}"):
            img_path = os.path.join(split_src, img_info['file_name'])
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            base_w = w // target_slices[0]
            base_h = h // target_slices[1]
            
            for r in range(target_slices[1]):
                for c in range(target_slices[0]):
                    # Calculate coordinates with overlap
                    # x1: start at base, but pull back by overlap if not the first column
                    # x2: end at base, but push forward by overlap if not the last column
                    x1 = max(0, c * base_w - (ov_w if c > 0 else 0))
                    y1 = max(0, r * base_h - (ov_h if r > 0 else 0))
                    x2 = min(w, (c + 1) * base_w + (ov_w if c < target_slices[0]-1 else 0))
                    y2 = min(h, (r + 1) * base_h + (ov_h if r < target_slices[1]-1 else 0))
                    
                    tile = img[y1:y2, x1:x2]
                    tile_h, tile_w = tile.shape[:2]
                    tile_name = f"{img_id_count}_{img_info['file_name']}"
                    cv2.imwrite(os.path.join(split_dst, tile_name), tile)
                    
                    new_images.append({
                        "id": img_id_count,
                        "file_name": tile_name,
                        "width": tile_w, 
                        "height": tile_h
                    })

                    for ann in coco['annotations']:
                        if ann['image_id'] == img_info['id']:
                            ax, ay, aw, ah = ann['bbox']
                            
                            if ax < x2 and ax + aw > x1 and ay < y2 and ay + ah > y1:
                                nx = max(0, ax - x1)
                                ny = max(0, ay - y1)
                                nw = min(ax + aw, x2) - max(ax, x1)
                                nh = min(ay + ah, y2) - max(ay, y1)
                                
                                if nw > 2 and nh > 2:
                                    new_ann = ann.copy()
                                    new_ann.update({
                                        "id": ann_id_count,
                                        "image_id": img_id_count,
                                        "bbox": [nx, ny, nw, nh],
                                        "area": nw * nh
                                    })
                                    
                                    if "segmentation" in ann:
                                        new_seg = []
                                        for poly in ann['segmentation']:
                                            shifted_poly = [(p - x1 if i % 2 == 0 else p - y1) 
                                                           for i, p in enumerate(poly)]
                                            new_seg.append(shifted_poly)
                                        new_ann["segmentation"] = new_seg
                                    
                                    new_anns.append(new_ann)
                                    ann_id_count += 1
                    img_id_count += 1

        with open(os.path.join(split_dst, "_annotations.coco.json"), 'w') as f:
            json.dump({
                "info": coco.get("info", {}),
                "categories": coco['categories'],
                "images": new_images,
                "annotations": new_anns
            }, f)
            
    return new_dir

# ---SLICING IMAGE FOR PREDICTIONS ----
class SlicedDetections:
    def __init__(self, boxes, scores, masks, class_ids):
        self.xyxy = np.array(boxes)
        self.confidence = np.array(scores)
        self.mask = masks # List of numpy arrays
        self.class_id = np.array(class_ids)

def predict_sliced_custom(model, img_np, target_slices, overlap_px, conf_thresh=0.2):
    # model.remove_optimized_model() # already exist in optimize_for_inference(); checked on rfdetr version 1.6.0
    batch_size = target_slices[0]*target_slices[1]
    model.optimize_for_inference(batch_size = batch_size)
    h, w = img_np.shape[:2]
    
    overlap_w, overlap_h = overlap_px[0], overlap_px[1]
    slice_w, slice_h = w // target_slices[0], h // target_slices[1]
    
    slice_coords = [
        [0, 0, slice_w + overlap_w, slice_h + overlap_h],           # top left
        [slice_w - overlap_w, 0, w, slice_h + overlap_h],           # top right
        [0, slice_h - overlap_h, slice_w + overlap_w, h],           # bottom left
        [slice_w - overlap_w, slice_h - overlap_h, w, h]            # bottom right
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