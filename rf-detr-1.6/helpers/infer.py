# fucntions for inference
import os
import sys
import json
import random
from PIL import Image
import supervision as sv
from rfdetr.detr import RFDETR
import matplotlib.pyplot as plt

# to handle path redirection for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from helpers.annotators import *
from helpers.utilities import load_dataset_info, get_model_and_config

# ----- PROVIDE PREDICTED IMAGES --------
def run_inference(model, image_paths, output_dir, class_names, colors):
    annotated = []
    model.optimize_for_inference()
    
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        detections = model.predict(img, threshold=0.2)
        annotated.append(
            annotate_image(img, detections, class_names, colors)
        )

    save_grid(annotated, output_dir, subname="predicted_grid.png")

# ----- PROVIDE ANNOTATED IMAGES -----
def annotate_validation(dataset, image_paths, output_dir, class_names, colors):
    gt_annotated = []
    for img_path in image_paths:
        detections = dataset.annotations[img_path]
        img_rgb = Image.open(img_path).convert("RGB")
        
        gt_annotated.append(
            annotate_image(img_rgb, detections, class_names, colors)
        )
    
    save_grid(gt_annotated, output_dir, subname="ground_truth_grid.png")

# ---- SAVE COLLAGE OF IMAGES --------
def save_grid(annotated, output_dir, subname="grid.png"):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, img in zip(axes.flat, annotated):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, subname)
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")
    # plt.show()

# ---- MANAGE IMAGES FOR COLLAGE PREDICTIONS ------
def collage_predictions(dataset_dir, output_dir, model_type):
    dataset_info = load_dataset_info(dataset_dir)
    ModelClass, _ = get_model_and_config(model_type)
    
    best_weights = os.path.join(output_dir, "checkpoint_best_total.pth")
    test_dir = os.path.join(dataset_dir, "test")
    ann_test_path = os.path.join(test_dir, "_annotations.coco.json")

    if not os.path.exists(ann_test_path):
        raise Exception(f"Annotation file not found at {ann_test_path}")

    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=test_dir,
        annotations_path=ann_test_path,
        force_masks=True
    )

    all_paths = dataset.image_paths
    sampled_paths = random.sample(all_paths, min(16, len(all_paths)))

    annotate_validation(dataset, sampled_paths, output_dir, 
                        dataset_info["class_names"], dataset_info["colors"])

    if os.path.exists(best_weights):
        infer_model = ModelClass(pretrain_weights=best_weights)
        run_inference(infer_model, sampled_paths, output_dir, 
                      dataset_info["class_names"], dataset_info["colors"])
    else:
        print(f"Weights not found at {best_weights}, skipping inference grid.")

# TO RUN INFER SCRIPT INDIVIDUALLY
# given that model has been trained and the metadata is generated
if __name__ == "__main__":
    metadata_path = "C:/Users/MachineLearning/Documents/Weight/rfdetr/trials/lens_station2.v22.combined_polar_unpolar2_sliced_overlap/001/training_metadata.json"
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    collage_predictions(
        dataset_dir=meta["dataset"]["dataset_dir"],
        output_dir=meta["dataset"]["output_dir"],
        model_type=meta["model"]["model_type"]
    )