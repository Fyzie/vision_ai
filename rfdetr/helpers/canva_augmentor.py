# Augmentor for LENS PROJECT
import os
import json
import cv2
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import math

# =========================================================
# SETTINGS
# =========================================================
# NEW: Range for training. For valid/test, it will use the max.
CANVAS_RANGE = (3, 10) 
MAX_RANGE = 20 # for valid data

GRID_COLS = 5
GRID_ROWS = 4
CELL_W, CELL_H = 240, 240 
TARGET_WIDTH = 1024

ROT_ANGLE_DEG = 20
SCALE_MIN = 0.8
SCALE_MAX = 1.2
FLIP_H_PROB = 0.5
FLIP_V_PROB = 0.5
MOTION_BLUR_PROB = 0.3
EXCLUDE_NAMES = ["incomplete", "missing", "rough_surface", "shining"] 

# ... [HELPER FUNCTIONS REMAIN THE SAME: mask_to_polygons, random_brightness, etc.] ...
def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3: continue
        polygons.append(contour.flatten().tolist())
    return polygons

def random_brightness_contrast(img, brightness=0.2, contrast=0.2):
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta = 255 * random.uniform(-brightness, brightness)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def random_saturation_hue(img, saturation=0.2, hue=0.1):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= 1 + random.uniform(-saturation, saturation)
    hsv[...,0] += random.uniform(-hue*180, hue*180)
    hsv[...,0] = np.clip(hsv[...,0], 0, 179)
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def elastic_transform_pair(img, masks, alpha=30, sigma=6):
    h, w = img.shape[:2]
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    new_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    new_masks = [cv2.remap(m, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for m in masks]
    return new_img, new_masks

def random_motion_blur(img, max_kernel_size=7):
    size = random.randint(3, max_kernel_size)
    kernel = np.zeros((size, size))
    direction = random.choice(['h', 'v', 'd1', 'd2'])
    if direction == 'h': kernel[int((size - 1) / 2), :] = np.ones(size)
    elif direction == 'v': kernel[:, int((size - 1) / 2)] = np.ones(size)
    elif direction == 'd1': np.fill_diagonal(kernel, 1)
    else: np.fill_diagonal(np.flipud(kernel), 1)
    kernel /= size
    return cv2.filter2D(img, -1, kernel)

def unroll_image_and_masks(img, masks, out_w, out_h):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    angle_offset = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D(center, angle_offset, 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    masks_rot = [cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST) for m in masks]
    dsize = (int(radius), int(out_w))
    unrolled_img = cv2.warpPolar(img_rot, dsize, center, int(radius), cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)
    unrolled_img = cv2.rotate(unrolled_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    unrolled_img = cv2.resize(unrolled_img, (out_w, out_h))
    new_masks = []
    for m in masks_rot:
        um = cv2.warpPolar(m, dsize, center, int(radius), cv2.WARP_POLAR_LINEAR + cv2.INTER_NEAREST)
        um = cv2.rotate(um, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_masks.append(cv2.resize(um, (out_w, out_h), interpolation=cv2.INTER_NEAREST))
    return unrolled_img, new_masks

# =========================================================
# CORE PROCESSING
# =========================================================
def process_split(folder_path, out_folder, is_train=False):
    ann_path = os.path.join(folder_path, "_annotations.coco.json")
    if not os.path.exists(ann_path): return
    with open(ann_path, 'r') as f: coco = json.load(f)

    exclude_ids = [cat['id'] for cat in coco['categories'] if cat['name'] in EXCLUDE_NAMES]
    new_categories = [cat for cat in coco['categories'] if cat['id'] not in exclude_ids]
    
    anns_by_img = {}
    for a in coco['annotations']:
        if a['category_id'] not in exclude_ids:
            anns_by_img.setdefault(a['image_id'], []).append(a)

    valid_images = [img for img in coco["images"] if img["id"] in anns_by_img]
    os.makedirs(out_folder, exist_ok=True)
    
    new_images, new_anns = [], []
    img_id_ptr, ann_id_ptr = 1, 1

    configs = [
        {"mode": "std", "aug": False, "suffix": ""},
        {"mode": "std", "aug": True, "suffix": "_aug"},
        # {"mode": "polar", "aug": False, "suffix": "_polar"},
        # {"mode": "polar", "aug": True, "suffix": "_polar_aug"}
    ] if is_train else [{"mode": "std", "aug": False, "suffix": ""}]

    canvas_w = GRID_COLS * CELL_W
    canvas_h = GRID_ROWS * CELL_H
    scale_factor = TARGET_WIDTH / canvas_w
    final_h = int(canvas_h * scale_factor)

    for cfg in configs:
        random.shuffle(valid_images)
        
        ptr = 0
        c_idx = 0
        pbar = tqdm(total=len(valid_images), desc=f"Pass: {cfg['mode']}{cfg['suffix']}")
        
        while ptr < len(valid_images):
            # 1. Determine random number of images for THIS canvas
            if is_train:
                num_to_take = random.randint(CANVAS_RANGE[0], CANVAS_RANGE[1])
            else:
                num_to_take = MAX_RANGE # CANVAS_RANGE[1] # Use fixed max for validation consistency
            
            chosen = valid_images[ptr : ptr + num_to_take]
            ptr += len(chosen)
            pbar.update(len(chosen))
            
            canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
            c_name = f"canvas_{cfg['mode']}_{c_idx+1}{cfg['suffix']}.png"
            c_idx += 1
            
            # 2. Pick random slots out of 20
            grid_indices = list(range(20))
            random.shuffle(grid_indices)
            assigned_slots = grid_indices[:len(chosen)]

            for i, info in enumerate(chosen):
                src_img = cv2.imread(os.path.join(folder_path, info["file_name"]))
                if src_img is None: continue
                
                img_anns = anns_by_img[info['id']]
                orig_masks = []
                for a in img_anns:
                    m = np.zeros(src_img.shape[:2], np.uint8)
                    if "segmentation" in a and a["segmentation"]:
                        for poly in a["segmentation"]:
                            cv2.fillPoly(m, [np.array(poly).reshape(-1, 2).astype(np.int32)], 255)
                    else:
                        x, y, bw, bh = map(int, a["bbox"])
                        m[y:y+bh, x:x+bw] = 255
                    orig_masks.append(m)

                if cfg["mode"] == "polar":
                    cell_img, cell_masks = unroll_image_and_masks(src_img, orig_masks, CELL_W, CELL_H)
                else:
                    cell_img = cv2.resize(src_img, (CELL_W, CELL_H))
                    cell_masks = [cv2.resize(m, (CELL_W, CELL_H), interpolation=cv2.INTER_NEAREST) for m in orig_masks]

                if cfg["aug"]:
                    if random.random() < FLIP_H_PROB:
                        cell_img = cv2.flip(cell_img, 1); cell_masks = [cv2.flip(m, 1) for m in cell_masks]
                    cell_img = random_brightness_contrast(cell_img)
                    cell_img = random_saturation_hue(cell_img)
                    cell_img, cell_masks = elastic_transform_pair(cell_img, cell_masks)
                    if random.random() < MOTION_BLUR_PROB:
                        cell_img = random_motion_blur(cell_img, max_kernel_size=9)

                slot_idx = assigned_slots[i]
                row, col = slot_idx // GRID_COLS, slot_idx % GRID_COLS
                x_off, y_off = col * CELL_W, row * CELL_H
                canvas[y_off:y_off+CELL_H, x_off:x_off+CELL_W] = cell_img
                
                for idx, a in enumerate(img_anns):
                    m_final = cell_masks[idx]
                    ys, xs = np.where(m_final > 0)
                    if len(xs) < 10: continue

                    sx0, sy0, sx1, sy1 = xs.min(), ys.min(), xs.max(), ys.max()
                    polys = mask_to_polygons(m_final)
                    
                    for poly in polys:
                        shifted_scaled_poly = []
                        for p_idx, val in enumerate(poly):
                            if p_idx % 2 == 0:
                                shifted_scaled_poly.append(float((val + x_off) * scale_factor))
                            else:
                                shifted_scaled_poly.append(float((val + y_off) * scale_factor))

                        scaled_bbox = [
                            float((sx0 + x_off) * scale_factor),
                            float((sy0 + y_off) * scale_factor),
                            float((sx1 - sx0 + 1) * scale_factor),
                            float((sy1 - sy0 + 1) * scale_factor)
                        ]

                        new_anns.append({
                            "id": ann_id_ptr, 
                            "image_id": img_id_ptr, 
                            "category_id": a['category_id'],
                            "bbox": scaled_bbox,
                            "area": float(scaled_bbox[2] * scaled_bbox[3]), 
                            "segmentation": [shifted_scaled_poly], 
                            "iscrowd": 0
                        })
                        ann_id_ptr += 1

            final_canvas = cv2.resize(canvas, (TARGET_WIDTH, final_h), interpolation=cv2.INTER_AREA)
            new_images.append({"id": img_id_ptr, "file_name": c_name, "width": TARGET_WIDTH, "height": final_h})
            cv2.imwrite(os.path.join(out_folder, c_name), final_canvas)
            img_id_ptr += 1
        pbar.close()

    with open(os.path.join(out_folder, "_annotations.coco.json"), "w") as f:
        json.dump({"info": coco.get("info", {}), "licenses": coco.get("licenses", []), 
                   "images": new_images, "annotations": new_anns, "categories": new_categories}, f, indent=2)

def batch_run(dataset_dir):
    out = dataset_dir + ".combined_polar_unpolar"
    for split in ["train", "valid", "test"]:
        src = os.path.join(dataset_dir, split)
        if os.path.exists(src):
            print(f"\n--- Processing {split} ---")
            process_split(src, os.path.join(out, split), is_train=(split=="train"))

    return out

if __name__ == "__main__":
    out = batch_run("D:/Pytorch Projects/work/lens/data/lens_station3-15")