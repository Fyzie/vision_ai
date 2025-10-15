import os
import io
import cv2
import json
import random
import datetime
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from tabulate import tabulate
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from contextlib import redirect_stdout

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn.functional as torch_F

from torchvision import ops

from utils.annotators import mask_annotator, box_annotator, label_annotator

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, model_type='maskrcnn', folder_type="train", transforms=None, verbose=True):
        self.root           = root
        self.transforms     = transforms
        self.model_type     = model_type
        self.folder_type    = folder_type

        with io.StringIO() as buffer, redirect_stdout(buffer):
            self.coco       = COCO(annotation_file)

        # Extract categories, remove id: 0 if it exists, and assign id: 0 as background
        self.categories     = self.coco.loadCats(self.coco.getCatIds())
        self.categories     = [{"id": 0, "name": "background"}] + [cat for cat in self.categories if cat["id"] != 0]
        self.class_names    = [category['name'] for category in self.categories]
        self.num_classes    = len(self.class_names)

        # Create category mapping with background as id: 0
        self.category_mapping = {category['id']: idx for idx, category in enumerate(self.categories)}

        self.ids = [
            img_id for img_id in tqdm(self.coco.imgs.keys(), desc=f"Validating {folder_type} image paths", ncols=100)
            if os.path.exists(os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name']))
        ]

        print(f"\n[{self.folder_type.capitalize()} Folder]")
        print(f"Total valid images found: {len(self.ids)}")
        print(f"Detected classes (including background): {self.class_names} (Total: {self.num_classes})")
        print()

    def __getitem__(self, index):
        img_id      = self.ids[index]
        img_info    = self.coco.loadImgs(img_id)[0]
        img_path    = os.path.join(self.root, img_info['file_name'])

        img         = Image.open(img_path).convert("RGB")

        ann_ids     = self.coco.getAnnIds(imgIds=img_id)
        anns        = self.coco.loadAnns(ann_ids)

        boxes   = []
        masks   = []
        labels  = []

        if anns:
            for ann in anns:
                xmin, ymin, w, h = ann['bbox']
                if w > 0 and h > 0:
                    boxes.append([xmin, ymin, xmin + w, ymin + h])
                    labels.append(self.category_mapping.get(ann['category_id'], 0))
                    if self.model_type == 'maskrcnn':
                        masks.append(self.coco.annToMask(ann))

        boxes   = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels  = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        if self.model_type == 'maskrcnn':
            masks   = torch.as_tensor(np.array(masks), dtype=torch.uint8) if masks else torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)

        target  = {"boxes": boxes, "labels": labels}
        if self.model_type == "maskrcnn":
            target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])
        
        if self.transforms:
            # Albumentations needs numpy
            img_np = np.array(img)

            # Prepare dict for Albumentations
            transform_input = {
                "image": img_np,
                "bboxes": boxes.numpy().tolist(),
                "labels": labels.numpy().tolist()
            }
            if self.model_type == "maskrcnn":
                transform_input["masks"] = masks.numpy()

            transformed = self.transforms(**transform_input)

            img = transformed["image"]  # already tensor from ToTensorV2
            boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.as_tensor(transformed["labels"], dtype=torch.int64)
            if self.model_type == "maskrcnn":
                masks = torch.as_tensor(transformed["masks"], dtype=torch.uint8)

        # rebuild target after transform
        target = {"boxes": boxes, "labels": labels}
        if self.model_type == "maskrcnn":
            target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])


        return img, target

    def __len__(self):
        return len(self.ids)
    
class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, mode='min', verbose=True, path=None):
        """
        Early stops the training if validation metric does not improve after a given patience.
        :param patience: Number of epochs to wait after last improvement.
        :param delta: Minimum change to qualify as improvement.
        :param mode: 'min' for metrics like loss, 'max' for metrics like accuracy/F1.
        :param path: Base path to save checkpoints (will append epoch dynamically).
        """
        self.patience       = patience
        self.delta          = delta
        self.mode           = mode
        self.verbose        = verbose
        self.best_metric    = None
        self.best_epoch     = None
        self.counter        = 0
        self.early_stop     = False
        self.base_path      = path or "checkpoint.pth"
        self.previous_checkpoint_path = ""

    def __call__(self, epoch_metric, model, epoch):
        if self.best_metric is None or \
           (self.mode == 'min' and epoch_metric < self.best_metric - self.delta) or \
           (self.mode == 'max' and epoch_metric > self.best_metric + self.delta):
            self.best_metric = epoch_metric
            self.best_epoch = epoch
            self.save_checkpoint(model, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}. Best metrics at epoch {self.best_epoch} with {self.best_metric:.4f} ")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, epoch):
        """
        Save a checkpoint with a dynamic filename based on the current epoch.
        """
        if os.path.exists(self.previous_checkpoint_path):
            os.remove(self.previous_checkpoint_path)

        checkpoint_path = os.path.join(self.base_path, f"checkpoint_epoch_{epoch + 1}.pth")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, checkpoint_path)

        self.previous_checkpoint_path = checkpoint_path

class Metadata:
    """
    Manage metadata of the training for inferencing uses
    """
    def __init__(self, metadata_path, **kwargs):
        self.metadata_path = metadata_path
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_attribute(self, key, value):
        setattr(self, key, value)

    def save(self):
        metadata_dict = {k: v for k, v in self.__dict__.items() if k != "metadata_path"}
        
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
        
        print(f"\nMetadata saved at {self.metadata_path}")

#---------------------------------------- DATA AND MODEL LOADING FUNCTIONS ----------------------------------------#


def get_transform(max_h, max_w):
    transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=max_h,
                min_width=max_w,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0, 
                fill_mask=0,
                p=1.0
            ),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        )
    )
    return transform

def collate_fn(batch):
    filtered_batch = [data for data in batch if data is not None]
    if len(filtered_batch) < len(batch):
        print(f"Skipped {len(batch) - len(filtered_batch)} invalid samples.")
    return tuple(zip(*filtered_batch))

def get_max_dims(img_root, coco):
    max_w, max_h = 0, 0
    for img_id in coco.imgs.keys():
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_root, img_info['file_name'])
        if os.path.exists(img_path):
            with Image.open(img_path) as im:
                w, h = im.size
                max_w = max(max_w, w)
                max_h = max(max_h, h)
    return max_w, max_h

def find_min_max_bbox_sizes(dataset):
    """
    Identify min and max of object-of-interests within the training dataset
    """
    min_width, min_height = float("inf"), float("inf")
    max_width, max_height = 0, 0

    for item in dataset.coco.anns.values():
        x, y, width, height = item["bbox"]

        min_width   = min(min_width, width)
        min_height  = min(min_height, height)
        max_width   = max(max_width, width)
        max_height  = max(max_height, height)

    return min_width, min_height, max_width, max_height

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

#---------------------------------------- MODEL EVALUATION FUNCTIONS ----------------------------------------#

def compute_advanced_metrics(epoch, targets, outputs, val_dataset, track_segment, debug_img_path, model_type='maskrcnn', iou_threshold=0.5, conf_threshold=0.5):
    """
    Evaluate bbox- and mask-level metrics (class location in terms of object-level and pixel-wise)
    """

    classwise_metrics   = {}
    all_mask_ious       = []

    for target, output in zip(targets, outputs):
        gt_boxes, gt_labels = target['boxes'], target['labels']
        pred_boxes, pred_labels, pred_scores = output['boxes'], output['labels'], output['scores']

        if model_type == 'maskrcnn':
            gt_masks = target['masks']
            pred_masks = output['masks']
        else:
            gt_masks = pred_masks = None

        keep = pred_scores > conf_threshold
        pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
        if model_type == 'maskrcnn':
            pred_masks = pred_masks[keep]

        unique_classes = torch.cat([gt_labels, pred_labels]).unique()

        for cls in unique_classes:
            cls_id = int(cls.item())
            gt_cls_boxes = gt_boxes[gt_labels == cls]
            pred_cls_boxes = pred_boxes[pred_labels == cls]

            if cls_id not in classwise_metrics:
                classwise_metrics[cls_id] = {'TP': 0, 'FP': 0, 'FN': 0, 'IoUs': []}

            if len(gt_cls_boxes) == 0 and len(pred_cls_boxes) == 0:
                continue
            if len(pred_cls_boxes) == 0:
                classwise_metrics[cls_id]['FN'] += len(gt_cls_boxes)
                continue
            if len(gt_cls_boxes) == 0:
                classwise_metrics[cls_id]['FP'] += len(pred_cls_boxes)
                continue

            ious = ops.box_iou(pred_cls_boxes, gt_cls_boxes)
            matched_gt, matched_pred = set(), set()
            TP = 0
            matched_pairs = []

            for pred_idx in range(len(pred_cls_boxes)):
                gt_idx = torch.argmax(ious[pred_idx]).item()
                if ious[pred_idx, gt_idx] > iou_threshold and gt_idx not in matched_gt:
                    TP += 1
                    matched_pairs.append((pred_idx, gt_idx))
                    matched_pred.add(pred_idx)
                    matched_gt.add(gt_idx)

            FP = len(pred_cls_boxes) - TP
            FN = len(gt_cls_boxes) - TP

            def save_debug_masks(pred_masks, gt_masks, epoch, cls):
                os.makedirs(debug_img_path, exist_ok=True)
                
                num_samples = len(pred_masks)
                fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

                if num_samples == 1:
                    axs = [axs]

                for idx in range(num_samples):
                    pred_mask = pred_masks[idx]
                    gt_mask   = gt_masks[idx]

                    axs[idx][0].imshow(pred_mask, cmap='gray')
                    axs[idx][0].set_title(f"Predicted Mask #{idx}")

                    axs[idx][1].imshow(gt_mask, cmap='gray')
                    axs[idx][1].set_title(f"Ground Truth Mask #{idx}")

                    axs[idx][2].imshow(gt_mask, cmap='Blues', alpha=0.5)
                    axs[idx][2].imshow(pred_mask, cmap='Reds', alpha=0.5)
                    axs[idx][2].set_title(f"Overlay #{idx} (GT: Blue, Pred: Red)")

                    for col in range(3):
                        axs[idx][col].axis("off")

                fig.tight_layout()
                save_name = f"mask_debug_epoch{epoch}_cls{cls}.png"
                save_path = os.path.join(debug_img_path, save_name)
                plt.savefig(save_path)
                plt.close(fig)

            matched_mask_ious = []
            debug_preds = []
            debug_gts   = []

            if model_type == 'maskrcnn':
                gt_cls_mask   = gt_masks[gt_labels == cls]
                pred_cls_mask = pred_masks[pred_labels == cls]

                for idx, (pred_idx, gt_idx) in enumerate(matched_pairs):
                    pred_mask    = (pred_cls_mask[pred_idx].squeeze(0) > 0.5).float()
                    gt_mask      = (gt_cls_mask[gt_idx] > 0.5).float()
                    intersection = (pred_mask * gt_mask).sum().float()
                    union        = (pred_mask + gt_mask - pred_mask * gt_mask).sum().float()
                    mask_iou     = (intersection / (union + 1e-6)).item()
                    matched_mask_ious.append(mask_iou)

                    if track_segment["save"] and idx < track_segment["limit"]:
                        debug_preds.append(pred_mask.cpu().numpy())
                        debug_gts.append(gt_mask.cpu().numpy())

                if track_segment["save"] and debug_preds:
                    save_debug_masks(debug_preds, debug_gts, epoch=epoch, cls=cls_id)

            classwise_metrics[cls_id]['TP'] += TP
            classwise_metrics[cls_id]['FP'] += FP
            classwise_metrics[cls_id]['FN'] += FN
            if model_type == 'maskrcnn':
                classwise_metrics[cls_id]['IoUs'].extend(matched_mask_ious)
                all_mask_ious.extend(matched_mask_ious)

    per_class_results = {}
    for cls_id, m in classwise_metrics.items():
        TP, FP, FN  = m['TP'], m['FP'], m['FN']
        precision   = TP / (TP + FP) if (TP + FP) else 0.0
        recall      = TP / (TP + FN) if (TP + FN) else 0.0
        f1          = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        avg_iou     = np.mean(m['IoUs']) if (model_type == 'maskrcnn' and m['IoUs']) else 0.0
        cls_name    = val_dataset.class_names[cls_id]

        per_class_results[cls_name] = {
            'bbox_precision': precision,
            'bbox_recall'   : recall,
            'bbox_f1_score' : f1,
            'mask_iou'      : avg_iou
        }

    overall = {
        'avg_precision' : np.mean([m['bbox_precision'] for m in per_class_results.values()]) if per_class_results else 0.0,
        'avg_recall'    : np.mean([m['bbox_recall'] for m in per_class_results.values()]) if per_class_results else 0.0,
        'avg_f1_score'  : np.mean([m['bbox_f1_score'] for m in per_class_results.values()]) if per_class_results else 0.0,
        'avg_iou'       : np.mean(all_mask_ious) if (model_type == 'maskrcnn' and all_mask_ious) else 0.0
    }

    return per_class_results, overall

def evaluate_model(epoch, model, val_dataset, val_loader, track_segment, debug_img_path, device, model_type='maskrcnn'):
    """
    Evaluate class-level metrics (class presence with image regardless their location, number)
    """
    model.eval()

    total_metrics = {'avg_precision': 0.0, 'avg_recall': 0.0, 'avg_f1_score': 0.0, 'avg_iou': 0.0}
    classwise_iou = {
        c: {"bbox_precision": 0.0, "bbox_recall": 0.0, "bbox_f1_score": 0.0, "mask_iou": 0.0}
        for c in val_dataset.class_names if c not in {"none", "background"}
    }
    raw_class_counts    = {c: 0 for c in val_dataset.class_names if c != "none"}
    confusion_metrics   = {c: {"TP": 0, "FP": 0, "FN": 0} for c in val_dataset.class_names if c != "none"}

    num_batches = 0

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Evaluating", ncols=100):
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(imgs)

            for i, output in enumerate(outputs):
                gt_labels = targets[i]['labels'].cpu().numpy()

                if len(gt_labels) == 0:
                    for pred_label in output['labels'].cpu().numpy():
                        pred_name = val_dataset.class_names[pred_label]
                        confusion_metrics[pred_name]["FP"] += 1
                    continue

                scores = output['scores'].cpu()

                high_conf_ids = torch.tensor(
                    [idx for idx, score in enumerate(scores) if score > 0.5],
                    dtype=torch.long
                )
                predicted_labels = output['labels'][high_conf_ids].cpu().numpy()

                for true_label in gt_labels:
                    raw_class_counts[val_dataset.class_names[true_label]] += 1

                matched = []
                for pred in predicted_labels:
                    pred_name = val_dataset.class_names[pred]
                    if pred in gt_labels:
                        confusion_metrics[pred_name]["TP"] += 1
                        matched.append(pred)
                    else:
                        confusion_metrics[pred_name]["FP"] += 1

                for true_label in gt_labels:
                    if true_label not in matched:
                        confusion_metrics[val_dataset.class_names[true_label]]["FN"] += 1

            per_class_iou, overall_iou = compute_advanced_metrics(epoch, targets, outputs, val_dataset, track_segment, debug_img_path, model_type=model_type)
            for k in total_metrics:
                total_metrics[k] += overall_iou[k]
            for cls_name in classwise_iou:
                if cls_name in per_class_iou:
                    for met in classwise_iou[cls_name]:
                        classwise_iou[cls_name][met] += per_class_iou[cls_name][met]

            num_batches += 1

    averaged_metrics        = {k: v / num_batches for k, v in total_metrics.items()}
    averaged_iou_by_class   = {
        cls: {k: v / num_batches for k, v in metrics.items()}
        for cls, metrics in classwise_iou.items()
    }

    total_TP = total_FP = total_FN = 0
    final_class_metrics = {}
    f1_by_class         = {}

    for cls_name, metrics in confusion_metrics.items():
        TP, FP, FN = metrics["TP"], metrics["FP"], metrics["FN"]
        total_TP += TP
        total_FP += FP
        total_FN += FN

        precision   = TP / (TP + FP) if (TP + FP) else 0
        recall      = TP / (TP + FN) if (TP + FN) else 0
        f1          = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        accuracy    = TP / (TP + FP + FN) if (TP + FP + FN) else 0

        final_class_metrics[cls_name] = {
            "class_precision"   : precision,
            "class_recall"      : recall,
            "class_f1_score"    : f1,
            "class_accuracy"    : accuracy,
            "TP"                : TP,
            "FP"                : FP,
            "FN"                : FN,
        }
        f1_by_class[cls_name]   = f1

    overall_precision   = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
    overall_recall      = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
    overall_f1          = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) else 0
    overall_accuracy    = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) else 0

    return {
        "overall": {
            "precision" : overall_precision,
            "recall"    : overall_recall,
            "f1_score"  : overall_f1,
            "accuracy"  : overall_accuracy,
        },
        "per_class_confusion": final_class_metrics,
        "per_class_iou" : averaged_iou_by_class,
        "iou_aggregated": averaged_metrics,
        "f1_by_class"   : f1_by_class,
        "valid_classes" : [c for c, count in raw_class_counts.items() if count > 0],
    }

#---------------------------------------- RESULT OUTPUT FUNCTIONS ----------------------------------------#

def create_segmentation_collages(model, val_dataset, device, class_colors, run_dir, best_model_path, model_type = 'maskrcnn', sample_ratio=1, threshold_overlay=0.5):
    """
    Create collages for predictions on validation dataset
    """
    print('[NOTE] Creating segmentation collages')
    print()

    sample_size      = max(1, int(len(val_dataset) * sample_ratio))
    sampled_indices  = random.sample(range(len(val_dataset)), sample_size)

    actual_images    = []
    predicted_images = []

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True), strict=False)

    model.cpu()
    with torch.no_grad():
        model.eval()

        for idx in sampled_indices:
            img, target = val_dataset[idx]
            img         = img.unsqueeze(0)
            output      = model(img)[0]

            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            actual_masks    = [mask for mask in target['masks']] if model_type == 'maskrcnn' else None
            actual_boxes    = target['boxes'].cpu().numpy()
            actual_colors   = [class_colors[label.item()] for label in target['labels']]
            actual_labels   = [val_dataset.class_names[label.item()] for label in target['labels']]

            img_actual_with_masks   = mask_annotator(img_np, actual_masks, actual_colors, alpha=0.5) if actual_masks is not None else img_np
            img_actual_with_boxes   = box_annotator(img_actual_with_masks, actual_boxes, actual_colors)
            img_actual_overlay      = label_annotator(img_actual_with_boxes, actual_boxes, actual_labels, actual_colors)

            high_conf_indices   = output['scores'] > threshold_overlay
            predicted_scores    = output['scores'][high_conf_indices]
            predicted_masks     = output['masks'][high_conf_indices] if model_type == 'maskrcnn' else None
            predicted_boxes     = output['boxes'][high_conf_indices].cpu().numpy()
            predicted_labels    = output['labels'][high_conf_indices]

            predicted_colors    = [class_colors[label.item()] for label in predicted_labels]
            predicted_labels    = [f"{val_dataset.class_names[label.item()]}: {score:.2f}" for label, score in zip(predicted_labels, predicted_scores)]

            def resize_and_binarize_masks(masks, image_size):
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)

                resized_masks   = torch_F.interpolate(masks, size=image_size, mode="bilinear", align_corners=False)
                binarized_masks = (resized_masks > 0.5).byte()
                return binarized_masks.squeeze(1)

            if predicted_masks is not None:
                resized_masks = resize_and_binarize_masks(predicted_masks, img_np.shape[:2])
                img_predicted_with_masks = mask_annotator(img_np, resized_masks, predicted_colors, alpha=0.5)
            else:
                img_predicted_with_masks = img_np

            img_predicted_with_boxes = box_annotator(img_predicted_with_masks, predicted_boxes, predicted_colors)
            img_predicted_overlay    = label_annotator(img_predicted_with_boxes, predicted_boxes, predicted_labels, predicted_colors)

            actual_images.append(torch.from_numpy(img_actual_overlay).permute(2, 0, 1) / 255.0)
            predicted_images.append(torch.from_numpy(img_predicted_overlay).permute(2, 0, 1) / 255.0)
    model.to(device)

    def save_collage(images, title, collage_name, cell_size=5, label_size=12):
        num_images  = len(images)
        cols        = 3
        rows        = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_size, rows * cell_size))
        for ax, img in zip(axes.flatten(), images):
            ax.imshow(img.permute(1, 2, 0))
            ax.axis("off")
        for ax in axes.flatten()[len(images):]:
            ax.axis("off")

        plt.suptitle(title, fontsize=label_size * 2)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        os.makedirs(run_dir, exist_ok=True)
        collage_path = os.path.join(run_dir, collage_name)
        plt.savefig(collage_path)
        plt.close()

    for i in range(0, len(actual_images), 9):
        actual_collage      = actual_images[i:i+9]
        predicted_collage   = predicted_images[i:i+9]
        save_collage(actual_collage, "Actual Images", f"{i//9 + 1}_image(actual).png")
        save_collage(predicted_collage, "Predicted Images", f"{i//9 + 1}_image(predicted).png")

def display_class_metrics_table(save_path, cm, rm, model_type='maskrcnn'):
    """
    Display per-class metrics in tabular format.
    """
    print('[NOTE] Displaying metrics table')
    print()

    table_data = []
    headers = ["Class Name", "Class Precision", "Class Recall", "Class F1 Score", "Class Accuracy", "Bbox Precision", "Bbox Recall", "Bbox F1 Score", "Mask IoU"]

    for class_name, metrics in cm.items():
        precision   = metrics["class_precision"]
        recall      = metrics["class_recall"]
        f1_score    = metrics["class_f1_score"]
        accuracy    = metrics["class_accuracy"]

        bbox_precision  = rm[class_name]["bbox_precision"]
        bbox_recall     = rm[class_name]["bbox_recall"]
        bbox_f1_score   = rm[class_name]["bbox_f1_score"]
        mask_iou        = rm[class_name]["mask_iou"]

        row = [
            class_name,
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1_score:.4f}",
            f"{accuracy:.4f}",
            f"{bbox_precision:.4f}",
            f"{bbox_recall:.4f}",
            f"{bbox_f1_score:.4f}",
        ]
        if model_type == 'maskrcnn':
            row.append(f"{mask_iou:.4f}")
        table_data.append(row)

    tabulated_str = tabulate(table_data, headers=headers, tablefmt="grid")

    print("\nPer-Class Metrics:")
    print(tabulated_str)

    with open(save_path, "w") as file:
        file.write(tabulated_str)

def plot_metrics(csv_file, run_dir):
    """
    Save metric plots/ graphs
    """
    print('[NOTE] Plotting metrics graphs')
    print()

    data = pd.read_csv(csv_file)

    metrics = [
        ('train_loss', 'val_loss', 'Loss Metrics'),
        ('train_classifier','val_classifier', 'Loss Classifier'),
        ('val_precision', 'Average Class Precision'),
        ('train_box_reg','val_box_reg', 'Loss Box Regression'), 
        ('val_recall', 'Average Class Recall'),
        ('train_mask','val_mask', 'Loss Mask'),
        ('val_f1_score', 'Average Class F1-Score'),
        ('lr', 'Learning Rate Per Epoch')
    ]

    os.makedirs(run_dir, exist_ok=True)

    for headers in metrics:
        plt.figure(figsize=(12, 6))
        for num in range(len(headers)-1):
            plt.plot(data['epoch'], data[headers[num]], label=headers[num].split("_")[0])
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()
        plt.title(headers[-1])
        plt.grid()

        plot_path = os.path.join(run_dir, f"{headers[-1].replace(' ', '_').lower()}.png")
        plt.savefig(plot_path)
        plt.close()

    print(f"Metrics plots saved to {run_dir}")

def save_confusion_matrix(per_class_metrics, class_names, save_path="confusion_matrix.png"):
    """
    Save class-level confusion matrix
    """
    print('[NOTE] Creating confusion matrix')
    print()

    labels          = class_names
    num_classes     = len(labels)
    
    conf_matrix     = np.zeros((num_classes, num_classes), dtype=int)
    
    class_to_index  = {name: i for i, name in enumerate(labels)}

    for class_name, metrics in per_class_metrics.items():
        true_idx = class_to_index[class_name]
        TP = metrics["TP"]
        FP = metrics["FP"]
        FN = metrics["FN"]

        conf_matrix[true_idx, true_idx] = TP

        if FP > 0:
            for _ in range(FP):
                pred_idx = np.random.choice([i for i in range(num_classes) if i != true_idx])
                conf_matrix[true_idx, pred_idx] += 1

        if FN > 0:
            for _ in range(FN):
                pred_idx = np.random.choice([i for i in range(num_classes) if i != true_idx])
                conf_matrix[pred_idx, true_idx] += 1

    conf_matrix_percentage  = conf_matrix.astype(float)
    row_sums                = conf_matrix_percentage.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_matrix_percentage = np.where(row_sums > 0, (conf_matrix_percentage / row_sums) * 100, 0)

    normalized_path = save_path.replace("confusion", "normalized_confusion")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Normalized Confusion Matrix")

    plt.savefig(normalized_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved as {save_path}")
