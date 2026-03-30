import os
import json
import rfdetr
import shutil
import colorsys
from pathlib import Path
from helpers.sahi import slice_dataset
from helpers.metadata import TrainingMetadata
from helpers.structures import TrainingSession
from helpers.custom_config import TrainingConfig, ModelConfig

from rich.console import Console
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

# ---- MONKEY-PATCH: MODIFY PTL COCOEVALCALLBACK TO INCLUDE THE METRICS TABLE SAVING -------
def patched_print_metrics_tables(self, trainer, split, overall, per_class, save_mode='w'): # 'w' only last epoch table will be saved, 'a' to save all epoch tables
    # run original function of _print_metrics_tables within COCOEvalCallback
    COCOEvalCallback.original_print_metrics_tables(self, trainer, split, overall, per_class)

    # patch for extended COCOEvalCallback (saved on .txt)
    if not getattr(trainer, "is_global_zero", True):
        return

    output_dir = getattr(trainer, "default_root_dir", "logs")
    log_file = os.path.join(output_dir, f"metrics_tables_{split}.txt")
    
    capture_console = Console(width=120, force_terminal=False, color_system=None)
    with capture_console.capture() as capture:
        capture_console.print(self._render_overall_merged(split.capitalize(), overall))
        
        if per_class:
            from rich.table import Table
            t2 = Table(title=f"{split.capitalize()} - Per-class Metrics", show_header=True)
            t2.add_column("Class")
            t2.add_column("AP 50:95", justify="right")
            t2.add_column("AR", justify="right")
            t2.add_column("F1", justify="right")
            t2.add_column("Precision", justify="right")
            t2.add_column("Recall", justify="right")
            
            def _fmt(v): return "—" if (v != v or v < 0) else f"{v:.4f}"
            
            for row in per_class:
                t2.add_row(row["name"], _fmt(row["ap"]), _fmt(row["ar"]), 
                           _fmt(row["f1"]), _fmt(row["precision"]), _fmt(row["recall"]))
            capture_console.print(t2)

    table_output = capture.get()
    with open(log_file, save_mode, encoding="utf-8") as f:
        f.write(f"\n{'='*20} Epoch {trainer.current_epoch} {'='*20}\n")
        f.write(table_output)
        f.write("\n")

# ---- GET MODEL AND RESPECTIVE CONFIG WITHIN RFDETR LIBRARY-------
def get_model_and_config(model_type: str):
    try:
        model_class = getattr(rfdetr, model_type)
        from rfdetr import config as rf_configs
        config_class = getattr(rf_configs, f"{model_type}Config")
        return model_class, config_class
    except AttributeError:
        raise ImportError(f"Model type '{model_type}' not found.")

# ---- CREATE TEST DIR IF NONE -------
def check_test_dir(dataset_dir, source_folder_name="valid"):
    test_dir = os.path.join(dataset_dir, "test")
    source_dir = os.path.join(dataset_dir, source_folder_name)

    if not os.path.exists(test_dir):
        shutil.copytree(source_dir, test_dir)

# ---- GENERATE TRIAL RUN FOLDER -------
def get_next_run_number(output_path: str, width: int = 3) -> str:
    if not os.path.exists(output_path): return "1".zfill(width)
    run_nums = [int(n) for n in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, n)) and n.isdigit()]
    return str(max(run_nums) + 1 if run_nums else 1).zfill(width)

def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

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

def check_mode(config_dict, dataset_dir):
    mode = config_dict.get('mode', 'standard')
    
    if mode == "sahi":
        print("SAHI based training applied")
        target_slices = config_dict.get('target_slices', None)
        overlap_px = config_dict.get('overlap_px', None)
        if target_slices and overlap_px:
            dataset_dir = slice_dataset(
                dataset_dir, 
                target_slices=target_slices, 
                overlap_px=overlap_px, 
                output_name="_sliced_overlap"
            )
        else:
            raise Exception("Please declare 'target_slices' (rows, cols) and 'overlap_px' (x_overlap, y_overlap) on CONFIG")
        
    elif mode == "standard":
        print("Standard training applied")

    else:
        raise Exception("Training mode is not valid")

    return dataset_dir

def setup_training(config_dict):
    # ------ INITIALIZE ALL AVALIABLE DIR ---------
    dataset_dir = config_dict.get('dataset_dir', "")
    output_root = config_dict.get('output_dir', "results")

    # ------ CHECK TRAINING MODE - to renew dataset dir ---------
    dataset_dir = check_mode(config_dict, dataset_dir)
    
    dataset_name = os.path.basename(dataset_dir)
    base_results_path = os.path.join(output_root, dataset_name)
    run_num = get_next_run_number(base_results_path)
    session_dir = os.path.join(base_results_path, run_num)
    os.makedirs(session_dir, exist_ok=True)

    print(f"\n--------------- RESULTS SAVED AT ---------------\n{Path(session_dir).as_posix()}\n")

    # ---- COLLECT DATASET INFO ---------
    dataset_info = load_dataset_info(dataset_dir)
    
    # ---- LOAD RELEVANT CONFIGS ---------
    model_type = config_dict.get('model_type', 'RFDETRSegSmall')
    ModelClass, ConfigClass = get_model_and_config(model_type)
    
    train_cfg = TrainingConfig()
    # custom_arc = config_dict.get('custom_model', None) # will be further implemented in future
    custom_arc = None
    model_cfg = ModelConfig() if custom_arc is not None else None

    # ---- SAVE METADATA ------
    metadata = TrainingMetadata(dataset_dir=dataset_dir, output_dir=session_dir)
    metadata.set_dataset_info(**dataset_info)
    if custom_arc is not None:
        metadata.set_model_config(**model_cfg.__dict__)
    else:
        metadata.set_model_config(model_type=model_type)
    metadata.set_training_params(**train_cfg.__dict__)
    metadata.save()

    # ---- INITIALIZE MODEL ------
    if custom_arc is not None:
        rf_config = ConfigClass(num_classes=dataset_info["num_classes"], **model_cfg.__dict__)
        model = ModelClass(config=rf_config)

    else:
        model = ModelClass()

    # extending original COCOEvalCallback module to save metrics table
    if not hasattr(COCOEvalCallback, 'original_print_metrics_tables'):
        COCOEvalCallback.original_print_metrics_tables = COCOEvalCallback._print_metrics_tables
        COCOEvalCallback._print_metrics_tables = patched_print_metrics_tables

    # RETURN THIS SESSION's TRAINING SETUP
    return TrainingSession(
        model=model,
        train_cfg=train_cfg,
        dataset_dir=dataset_dir,
        session_dir=session_dir,
        dataset_info=dataset_info,
        model_type = model_type,
        model_class=ModelClass
    )