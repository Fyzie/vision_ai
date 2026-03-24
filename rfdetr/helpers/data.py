import os
import cv2
import json
import shutil
import colorsys
from tqdm import tqdm
from helpers.canva_augmentor import *

def check_test_dir(dataset_dir, source_folder_name="valid"):
    test_dir = os.path.join(dataset_dir, "test")
    source_dir = os.path.join(dataset_dir, source_folder_name)

    if not os.path.exists(test_dir):
        shutil.copytree(source_dir, test_dir)

def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_dataset_info(dataset_dir):
    annotation_path = os.path.join(
        dataset_dir, "train", "_annotations.coco.json"
    )

    with open(annotation_path, "r") as f:
        data = json.load(f)

    class_names = [c["name"] for c in data["categories"]]
    colors = dict(zip(class_names, generate_colors(len(class_names))))

    return {
        "annotation_path": annotation_path,
        "num_classes": len(class_names),
        "class_names": class_names,
        "colors": colors,
    }

def slice_dataset(src_dataset_dir, target_slices=(2, 2), overlap_px=(80, 0), output_name = "_sliced"):
    """
    Slices dataset with specific pixel overlap for width and height.
    overlap_px: (overlap_width, overlap_height) in pixels.
    """
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
            
            # Base dimensions without overlap
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

                    # Adjust annotations
                    for ann in coco['annotations']:
                        if ann['image_id'] == img_info['id']:
                            ax, ay, aw, ah = ann['bbox']
                            
                            # Intersection check
                            if ax < x2 and ax + aw > x1 and ay < y2 and ay + ah > y1:
                                # Shift to tile relative coordinates
                                nx = max(0, ax - x1)
                                ny = max(0, ay - y1)
                                # Clip width/height to tile boundaries
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
                                            # Shift every x (even index) and y (odd index)
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