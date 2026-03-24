import os
import json
import rfdetr
from collections import Counter
from helpers.gpu_monitor import *
from helpers.data import *
from helpers.infer import run_inference
from helpers.metadata import TrainingMetadata
from helpers.settings import ModelConfig, TrainingConfig
from rfdetr.util.misc import MetricLogger

def clean_str(self):
    whitelist = ['loss', 'class_error', 'loss_ce', 'loss_bbox', 'loss_giou', 'loss_mask_dice']
    loss_str = [f"{name}: {meter}" for name, meter in self.meters.items() if name in whitelist]
    return self.delimiter.join(loss_str)

MetricLogger.__str__ = clean_str

def get_next_run_number(output_path: str, width: int = 3) -> str:
    if not os.path.exists(output_path): return "1".zfill(width)
    run_nums = [int(n) for n in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, n)) and n.isdigit()]
    return str(max(run_nums) + 1 if run_nums else 1).zfill(width)

def analyze_dataset_density(annotation_path, num_queries_threshold=100):
    if not os.path.exists(annotation_path): return
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    counts = list(Counter([ann['image_id'] for ann in data['annotations']]).values())
    if not counts: return

    avg_objs, max_objs = sum(counts) / len(data['images']), max(counts)
    over_limit = sum(1 for c in counts if c > num_queries_threshold)

    print(f"\n{'='*40}\nDATASET DENSITY: Avg {avg_objs:.2f} | Max {max_objs} | Over Limit: {over_limit}\n{'='*40}")
    if over_limit > 0:
        print(f"WARNING: Consider increasing num_queries to at least {max_objs}.")

def get_model_and_config(model_type: str):
    """
    to retrieves the Model class and Config class based on string name.
    e.g.: RFDETRSegSmall -> RFDETRSegSmall, RFDETRSegSmallConfig
    """
    try:
        model_class = getattr(rfdetr, model_type)
        from rfdetr import config as rf_configs
        config_class = getattr(rf_configs, f"{model_type}Config")
        return model_class, config_class
    except AttributeError:
        raise ImportError(f"Model type '{model_type}' or its config is not supported or was not found in rfdetr.")

def execute_training_pipeline(dataset_dir, output_dir, experiment_name, model_type, pretrain_weights=None):
    dataset = load_dataset_info(dataset_dir)
    model_cfg, train_cfg = ModelConfig(), TrainingConfig()
    
    analyze_dataset_density(dataset["annotation_path"], num_queries_threshold=model_cfg.num_queries)

    ModelClass, ConfigClass = get_model_and_config(model_type)

    metadata = TrainingMetadata(dataset_dir=dataset_dir, output_dir=output_dir, experiment_name=experiment_name)
    metadata.set_dataset_info(num_classes=dataset["num_classes"], class_names=dataset["class_names"], 
                              colors=dataset["colors"], annotation_path=dataset["annotation_path"])
    metadata.set_model_config(model_type=model_type, **model_cfg.__dict__)
    metadata.set_training_params(**train_cfg.__dict__)
    metadata.save()

    check_gpu_vram(required_gb=8)

    config = ConfigClass(**model_cfg.__dict__, num_classes=dataset["num_classes"])
    model = ModelClass(config=config, pretrain_weights=pretrain_weights)

    model.train(dataset_dir=dataset_dir, output_dir=output_dir, **train_cfg.__dict__)
    cleanup_gpu_memory(model, verbose=True)

    best_weights = os.path.join(output_dir, "checkpoint_best_total.pth")
    if os.path.exists(best_weights):
        infer_model = ModelClass(pretrain_weights=best_weights)
        test_dir = os.path.join(dataset_dir, "test")
        if os.path.exists(test_dir):
            images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))]
            run_inference(infer_model, images, output_dir, dataset["class_names"], dataset["colors"])
    
    return best_weights

def run_experiment(
    experiment_name: str, 
    dataset_dir: str, 
    model_type: str,
    output_parent_dir: str = "", 
    mode: str = "standard",
    **kwargs
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    check_test_dir(dataset_dir)

    inference_batch = 1
    if mode == "sahi":
        target_slices = kwargs.get('target_slices', (2, 2))
        overlap_px = kwargs.get('overlap_px', (120, 0))
        dataset_dir = slice_dataset(
            dataset_dir, 
            target_slices=target_slices, 
            overlap_px=overlap_px, 
            output_name="_sliced_overlap"
        )
        inference_batch = target_slices[0] * target_slices[1]

    dataset_folder = os.path.basename(dataset_dir)
    base_results_path = os.path.join(output_parent_dir or script_dir, "results", dataset_folder)
    run_num = get_next_run_number(base_results_path)
    session_dir = os.path.join(base_results_path, run_num)
    os.makedirs(session_dir, exist_ok=True)

    execute_training_pipeline(dataset_dir, session_dir, experiment_name, model_type)


if __name__ == "__main__":
    CONFIG = {
        "mode": "standard", # Options: "standard" or "sahi"
        "model_type": "RFDETRSegSmall", # Options: RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, etc.
        "experiment_name": "rfdetr-station3-point-based",
        "dataset_dir": "D:/Pytorch Projects/work/lens/data/lens_station3-16.combined_polar_unpolar2",
        "output_parent_dir": "D:/Pytorch Projects/work/lens",
        "target_slices": (2, 2),
        "overlap_px": (120, 0),
    }

    run_experiment(**CONFIG)