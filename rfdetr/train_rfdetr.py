# train_rfdetr_local.py
import os
import shutil
import numpy as np
import random
import gc
import weakref
from PIL import Image
import matplotlib.pyplot as plt
import torch
import json

from rfdetr import RFDETRSegPreview
from rfdetr.config import RFDETRSegPreviewConfig

import supervision as sv

def check_gpu_vram(required_gb=8):
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Training will run on CPU.")
        return False

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    free = total - (reserved + allocated)

    print(f"[GPU] Model: {props.name}")
    print(f"[GPU] Total VRAM : {total:.2f} GB")
    print(f"[GPU] Free VRAM  : {free:.2f} GB")

    if free < required_gb:
        print(f"[WARNING] Expected at least {required_gb} GB free. You may OOM.")
        return False

    print("[GPU] VRAM check passed.")
    return True

def cleanup_gpu_memory(obj=None, verbose=False):
    if not torch.cuda.is_available():
        return

    def stats():
        return (
            torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.memory_reserved() / 1024**2,
        )

    torch.cuda.synchronize()

    if verbose:
        a, r = stats()
        print(f"[Cleanup Before] Alloc: {a:.2f} MB | Reserved: {r:.2f} MB")

    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        if ref() is not None and verbose:
            print("[WARNING] Object not fully deleted.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    if verbose:
        a, r = stats()
        print(f"[Cleanup After]  Alloc: {a:.2f} MB | Reserved: {r:.2f} MB")

def annotate_img(image, detections, classes=None):
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    if classes is not None:
        labels = [f"{classes[d]}: {c:.2f}" for c, d in zip(detections.confidence, detections.class_id)]
    else:
        labels = None

    overlay = box_annotator.annotate(
        scene=image,
        detections=detections
    )
    overlay = label_annotator.annotate(
        scene=overlay,
        detections=detections,
        labels = labels
    )

    return overlay

def check_test_dir(dataset_dir, source_folder_name="valid"):
    test_dir = os.path.join(dataset_dir, "test")
    source_dir = os.path.join(dataset_dir, source_folder_name)

    if not os.path.exists(test_dir):
        shutil.copytree(source_dir, test_dir)

#main
def main():

    dataset_dir = "/path/to/dataset"
    dataset_folder = os.path.basename(dataset_dir)
    output_dir = f"/path/to/output/folder"

    check_test_dir(dataset_dir)

    annotation_path = os.path.join(dataset_dir, 'train', '_annotations.coco.json')
    with open(annotation_path, 'r') as file:
        data = json.load(file)

    num_classes = len(data['categories'])
    print(f"Number of classes: {num_classes}")
        
    check_gpu_vram(required_gb=8)

    # configurations
    config = RFDETRSegPreviewConfig(
        encoder="dinov2_windowed_small",
        num_classes=num_classes,
        resolution=1024, # default:560
        num_queries=200, # default
        num_select=200, # default
        patch_size=12, # default
        dec_layers=4, # default
        num_windows=2, # default
        segmentation_head=True, # default
    )

    model = RFDETRSegPreview(config=config)

    model.train(
        dataset_dir=dataset_dir,
        epochs=50,
        batch_size=2,
        grad_accum_steps=4,
        lr=1e-4,
        lr_encoder=1e-4,
        early_stopping = True,
        early_stopping_patience = 10,
        early_stop = True,
        patience = 5,
        num_workers=0,
        multi_scale=True,
        expanded_scales=False,
        use_ema=True,
        output_dir=output_dir,
    )

    cleanup_gpu_memory(model, verbose=True)

    checkpoint_path = os.path.join(output_dir, "checkpoint_best_total.pth")
    model = RFDETRSegPreview(pretrain_weights=checkpoint_path)
    model.optimize_for_inference()

    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=os.path.join(dataset_dir, "test"),
        annotations_path=os.path.join(dataset_dir, "test", "_annotations.coco.json"),
        force_masks=True
    )

# show annotated images
    N = 16
    L = len(ds_test)
    annotated_images = []

    for i in random.sample(range(L), N):
        path, _, annotations = ds_test[i]
        image = Image.open(path).convert("RGB")
        detections = model.predict(image, threshold=0.5)

        if detections:
            boxes, masks, scores, class_ids = detections.xyxy, detections.mask, detections.confidence, detections.class_id

            print(f"Boxes: {boxes}")
            print(f"Masks: {masks}")
            print(f"Scores: {scores}")
            print(f"Class IDs: {class_ids}")

        annotated = annotate_img(
            image=image,
            detections=detections,
            classes={idx: c for idx, c in enumerate(ds_test.classes)}
        )
        annotated_images.append(annotated)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, img in zip(axes.flat, annotated_images):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "annotated_grid.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"(NOTE) Annotated grid saved to: {output_path}")

    plt.show()

if __name__ == "__main__":
    main()
