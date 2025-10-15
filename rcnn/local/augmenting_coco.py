import os
import json
import cv2
import shutil
import time
import numpy as np
import albumentations as A
from tqdm import tqdm

def polygons_to_mask(image_shape, polygons):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [pts], color=1)
    return mask

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.flatten().tolist()
        polygons.append(contour)
    return polygons

def clamp_bbox_to_image(bbox, img_h, img_w):
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    w = x2 - x
    h = y2 - y
    return [x, y, w, h] if w > 1 and h > 1 else None

def augment_loop(images, image_dir, annotations, num_augment_per_image, transform, output_dir, new_image_id, new_ann_id, augmented_images, augmented_annotations):
    for img_info in tqdm(images, desc="Augmenting"):
        img_id = img_info["id"]
        img_file = img_info["file_name"]
        img_path = os.path.join(image_dir, img_file)

        image = cv2.imread(img_path)
        if image is None:
            tqdm.write(f"Failed to load image {img_path}. Skipping.")
            continue

        annos_for_img = [ann for ann in annotations if ann["image_id"] == img_id]
        if len(annos_for_img) == 0:
            tqdm.write(f"No annotations for image_id={img_id}. Skipping.")
            continue

        bboxes = []
        masks = []
        category_ids = []

        for ann in annos_for_img:
            cat_id = ann["category_id"]
            bbox = ann["bbox"]  # COCO: [x, y, w, h]
            seg = ann.get("segmentation", [])

            if seg:
                mask = polygons_to_mask(image.shape, seg)
                if mask.sum() == 0:
                    continue
                masks.append(mask)

            bboxes.append(bbox)
            category_ids.append(cat_id)

        for _ in range(num_augment_per_image):
            try:
                transformed = transform(
                    image=image,
                    masks=masks if seg else [],
                    bboxes=bboxes,
                    category_ids=category_ids
                )
            except Exception as e:
                tqdm.write(f"Error transforming {img_file}: {e}")
                continue

            aug_image = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_masks = transformed["masks"] if seg else [None]*len(aug_bboxes)
            aug_cat_ids = transformed["category_ids"]

            h_img, w_img = aug_image.shape[:2]
            final_bboxes = []
            final_polygons = []
            final_cats = []

            for bb, mk, cid in zip(aug_bboxes, aug_masks, aug_cat_ids):
                if mk is not None:
                    new_polys = mask_to_polygons(mk) 
                    if len(new_polys) == 0:  
                        continue 
                    final_polygons.append(new_polys)


                clamped_bb = clamp_bbox_to_image(bb, h_img, w_img)
                if not clamped_bb:
                    continue

                final_bboxes.append(clamped_bb)
                final_cats.append(cid)

            if not final_bboxes:
                continue

            aug_img_name = f"aug_{os.path.splitext(img_file)[0]}_{time.time()}.jpg"
            aug_img_path = os.path.join(output_dir, aug_img_name)
            cv2.imwrite(aug_img_path, aug_image)

            new_img_dict = {
                "id": new_image_id,
                "file_name": aug_img_name,
                "width": w_img,
                "height": h_img,
                "license": img_info.get("license", 1),
                "date_captured": img_info.get("date_captured", "2024-01-01")
            }
            augmented_images.append(new_img_dict)

            if not final_polygons:
                final_polygons = [None]*len(final_bboxes)

            for bb, polygons_list, cid in zip(final_bboxes, final_polygons, final_cats): ########## final_polygons, poligon_list ###################################################################
                x, y, w, h = bb
                area_approx = w * h


                annotation_seg = [] 
                if polygons_list is not None:
                    for poly_coords in polygons_list: 
                        annotation_seg.append(poly_coords) 

                new_ann = {
                        "id": new_ann_id,
                        "image_id": new_image_id,
                        "category_id": cid,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(area_approx),
                    }
                if annotation_seg:
                    new_ann["segmentation"] = annotation_seg
                new_ann['iscrowd'] = 0
                augmented_annotations.append(new_ann)
                new_ann_id += 1

            new_image_id += 1

    return augmented_images, augmented_annotations, new_image_id, new_ann_id, transformed["replay"]

def get_train_transforms():
    transform = A.ReplayCompose(
        [
            # A.Resize(1024, 1024, p=1.0),
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
            A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR, p=1.0),
            ########################################## random scale implementation of object-of-interests ################################################## 
            # Additional: Random scale and padded to adapt various scales of object of interest 
            A.RandomScale(scale_limit=(0.5, -0.2),interpolation=cv2.INTER_LINEAR,mask_interpolation=cv2.INTER_NEAREST,area_for_downscale="image", p=0.4),
            # A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
            ################################################################################################################################################
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.ToGray(p=0.2),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.MotionBlur(blur_limit=(3, 5), p=0.3),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.8),
            A.ElasticTransform(alpha=100, sigma=50, p=0.5),
            A.Downscale(scale_range=(0.5, 0.75), interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR}, p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"]
        )
    )
    return transform

def get_valid_transforms():
    transform = A.ReplayCompose(
        [
            # A.Resize(1024, 1024, p=1.0),
            A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR, p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"]
        )
    )
    return transform

def augment_coco_dataset(data_to_augment, image_dir, annotation_path, output_dir, train_transform=get_train_transforms(), valid_transform=get_valid_transforms(), num_augs=2):
    transforms = [valid_transform, train_transform]
    with open(annotation_path, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, "_annotations.coco.json")
    shutil.copy(annotation_path, output_json_path)

    with open(output_json_path, "r") as f:
        new_coco_data = json.load(f)

    new_coco_data["images"] = []
    new_coco_data["annotations"] = []

    if len(images) == 0:
        print("No images found in JSON. Exiting.")
        return
    if len(annotations) == 0:
        print("No annotations found in JSON. Exiting.")
        return

    start_image_id = 0
    start_ann_id = 0

    augmented_images = []
    augmented_annotations = []

    replays = []
    for i, transform in enumerate(transforms):
      num_augment_per_image = num_augs if i != 0 and data_to_augment == "train" else 1
      print(f"Augmenting {data_to_augment} data with {num_augment_per_image} copies")
      augmented_images, augmented_annotations, start_image_id, start_ann_id, replay = augment_loop(images,
                                                                                           image_dir,
                                                                                           annotations,
                                                                                           num_augment_per_image,
                                                                                           transform,
                                                                                           output_dir,
                                                                                           start_image_id,
                                                                                           start_ann_id,
                                                                                           augmented_images,
                                                                                           augmented_annotations)
      replays.append(replay)
      if data_to_augment == "valid" or data_to_augment == "test":
        break
    new_coco_data["images"].extend(augmented_images)
    new_coco_data["annotations"].extend(augmented_annotations)

    with open(output_json_path, "w") as f:
        json.dump(new_coco_data, f, indent=2)

    def convert_numpy(obj):
      if isinstance(obj, np.ndarray):
          return obj.tolist()
      raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    json_file = os.path.join(os.path.dirname(output_json_path), "replay.json")
    with open(json_file, "a") as f:
      json.dump(replays, f, indent=4, default=convert_numpy)


    print()
    print(f"[NOTE] Augmentations completed for {data_to_augment} dataset")
    print(f"Augmented {data_to_augment} images saved at: {output_dir}")
    print(f"Augmented {data_to_augment} annotations saved at: {output_json_path}")
    print()

def main(
        data_type       = 'roboflow',                               # 'local' and 'roboflow'
        local_data      = None,
        roboflow_data   = None,                                 # roboflow_dataset().location or None
        result_path     = None,                                 
        data_dist       = {'train': 3, 'valid': 1, 'test': 1},
        train_transform = get_train_transforms(), 
        valid_transform = get_valid_transforms(),
):
    print()
    
    if data_type    == 'local':
        if local_data:
            print("[NOTE] Using Local dataset")
            print()
            dataset_path = local_data
        else:
            raise Exception("Please input your Local data path")

    elif data_type  == 'roboflow':
        if roboflow_data:
            print("[NOTE] Using Roboflow dataset")
            print()
            dataset_path = roboflow_data
        else:
            raise Exception("Please input your Roboflow data path")
    
    if result_path is None:
        raise Exception("Please input result path for Roboflow dataset")
    else:
        print(f"[NOTE] Your result will be at {result_path}")
        print()

    os.makedirs(result_path, exist_ok=True)
    os.chdir(result_path)

    if data_type == 'roboflow':
        dataset_name = os.path.basename(dataset_path.rstrip("/"))
        target_path = os.path.join(result_path, dataset_name)

        if not os.path.exists(target_path):
            print(f"[NOTE] Moving roboflow dataset from {dataset_path} to {target_path}")
            print()
            shutil.move(dataset_path, target_path)
        else:
            print(f"[NOTE] Dataset already exists at {target_path}")
            print()
            shutil.rmtree(dataset_path)
        dataset_path    = target_path
        
    
    base_name   = os.path.basename(dataset_path)
    version     = len([d for d in os.listdir(result_path) if d.startswith(f'{base_name}-augment')]) + 1

    for data_to_augment, num_augs in data_dist.items():
        IMAGE_DIR       = f"{dataset_path}/{data_to_augment}"
        ANNOTATION_PATH = f"{IMAGE_DIR}/_annotations.coco.json"
        OUTPUT_DIR      = f"{dataset_path}-augment.{version}/{data_to_augment}"

        if os.path.exists(IMAGE_DIR):
            augment_coco_dataset(
                data_to_augment,
                image_dir       = IMAGE_DIR,
                annotation_path = ANNOTATION_PATH,
                output_dir      = OUTPUT_DIR,
                train_transform = train_transform, 
                valid_transform = valid_transform,
                num_augs        = num_augs
            )

    print(f'[COMPLETE] Augmented dataset saved at: {os.path.dirname(OUTPUT_DIR)}')
    print(f'> Copy above dir for training <')
    print()

def roboflow_dataset():
    from roboflow import Roboflow
    rf = Roboflow(api_key="")
    project = rf.workspace("").project("")
    version = project.version(7)
    dataset = version.download("coco-segmentation")
                    
    return dataset

if __name__ == '__main__':
    main(
        data_type       = 'roboflow',                                                              
        # local_data      = r'',                                                             
        roboflow_data   = roboflow_dataset().location,                                             
        result_path     = r'',            
        data_dist       = {'train': 3, 'valid': 1, 'test': 1},                                 
        train_transform = get_train_transforms(), 
        valid_transform = get_valid_transforms(),
    )
