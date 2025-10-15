# training fasterrcnn and maskrcnn resnet

import os
import csv
import time
from tqdm import tqdm
import multiprocessing
from roboflow import Roboflow

import torch
from torch.utils.data import DataLoader

from utils.models import select_model
from utils.trains import (
    COCODataset, EarlyStopping, Metadata,
    get_max_dims, get_transform, collate_fn, find_min_max_bbox_sizes, generate_colors,
    evaluate_model, create_segmentation_collages, plot_metrics, display_class_metrics_table, save_confusion_matrix
)

#---------------------------------------- ROBOFLOW DATA ----------------------------------------#
def roboflow_dataset():
    rf = Roboflow(api_key="---")
    project = rf.workspace("---").project("---")
    version = project.version(1)
    dataset = version.download("coco-segmentation")

    return dataset
#---------------------------------------- TRAINING PARAMETERS ----------------------------------------#
def main(
    project_name        = "Project",
    dataset_location    = None,
    result_path         = None,
    resume              = False,
    model_path          = None,                                 # if want to resume model training
    num_workers         = 'default',
    num_epochs          = 10,
    early_stop          = True,
    patience            = 'default',                            # 'default': round(num_epochs * 0.2)
    batch_size          = 4,
    lr                  = 0.0001,                               # fixed
    weight_decay        = 0.0005,                               # overfit increase, underfit decrease
    optimizer_version   = 'simple',                             # simple(wd to all active params) or grouped(wd to only cov weights)
    scheduler_type      = None,
    track_segment       = {'save': True, 'limit' : 3},          # to debug 'mask generation' during evaluation phase
    model_type          = "maskrcnn",                           # "fasterrcnn: bbox" or "maskrcnn: instance segmentation"
    model_version       = '2',                                  # '1: custom' or '1.1: prepackaged' or '2: prepackaged'
    is_custom_anchor    = False,   
    # if is_custom_anchor == True
    anchor_sizes        = [(4,), (8,), (16,), (64,), (128,)],   # num of anchors follow num_feature_maps
    aspect_ratios       = [(0.33, 0.5, 1.0, 2.0, 3.0)],         # < 1: tall, > 1: wide     
    # backbone_name only applicable for model_version == '1: custom'
    backbone_name       = "resnet50",                           # from torchvision.models import resnet
    trainable_layers    = 3,                                    # from 1 to 5
    roboflow_data       = None,                                 # roboflow_dataset().location or None
):
    if patience == 'default':
        patience = round(num_epochs * 0.2)
    aspect_ratios = aspect_ratios * len(anchor_sizes)
#---------------------------------------- ALL PATHS INITIALIZATION ----------------------------------------#

    script_dir      = os.path.dirname(os.path.abspath(__file__))
    if result_path == None:
        result_path = os.path.join(script_dir, "results", project_name)
    else:
        result_path = os.path.join(result_path, "results", project_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Created directory: {result_path}")
    os.chdir(result_path)

    print(f'[NOTE] Your results will be saved at {result_path}')
    print()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    print()

    script_version          = os.path.basename(__file__).replace(".py","").split("_")[-1]
    dir                     = f"V{script_version}"

    os.makedirs(dir, exist_ok=True)
    run_dir                 = f"{dir}/run{len([d for d in os.listdir(dir) if d.startswith('run')]) + 1}"
    os.makedirs(run_dir, exist_ok=True)

    best_model_path         = os.path.join(run_dir, "best_model.pth")
    last_epoch_model_path   = os.path.join(run_dir, "last_epoch_model.pth")
    csv_file                = os.path.join(run_dir, "metrics.csv")
    metadata_path           = os.path.join(run_dir, "metadata.json")
    metrics_path            = os.path.join(run_dir, "evaluation.txt")
    cm_path                 = os.path.join(run_dir, "confusion_matrix.png")
    debug_img_path          = os.path.join(run_dir, "debug_imgs")

    train_folder            = "/train"
    train_annotation        = "/train/_annotations.coco.json"
    valid_folder            = "/valid"
    valid_annotation        = "/valid/_annotations.coco.json"

#---------------------------------------- DATASET LOADER ----------------------------------------#
    if roboflow_data is None and dataset_location is None:
        raise Exception('Please input either "dataset_location" for local or "roboflow_data"')
    elif roboflow_data is not None and dataset_location is not None:
        raise Exception('Please input ONLY either "dataset_location" for local or "roboflow_data"')
    else:
        if roboflow_data is None:
            print('[NOTE] Using Local dataset')
            print()
            dirroot = dataset_location
        else:
            print('[NOTE] Using Roboflow dataset')
            print()
            dirroot = roboflow_data

    if num_workers == 'default':
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Number of workers: {num_workers}")
    print()

    from pycocotools.coco import COCO
    coco_train = COCO(dirroot+train_annotation)
    max_w, max_h = get_max_dims(dirroot+train_folder, coco_train)

    train_dataset = COCODataset(
        root            = dirroot+train_folder,
        annotation_file = dirroot+train_annotation,
        model_type      = model_type,
        folder_type     = "train",
        transforms      = get_transform(max_h, max_w)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = num_workers > 0,
        persistent_workers = num_workers > 0,
        **({'prefetch_factor': 4} if num_workers > 0 else {}),
    )


    val_dataset = COCODataset(
        root            = dirroot+valid_folder,
        annotation_file = dirroot+valid_annotation,
        model_type      = model_type,
        folder_type     = "valid",
        transforms      = get_transform(max_h, max_w)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = num_workers > 0,
        persistent_workers = num_workers > 0,
        **({'prefetch_factor': 4} if num_workers > 0 else {}),
    )

    total_width, total_height = 0, 0
    for i in tqdm(range(len(train_dataset)), desc="Calculating avg image size", ncols=100):
        image, _ = train_dataset[i]
        _, h, w = image.shape
        total_width += w
        total_height += h

    w = int(total_width / len(train_dataset))
    h = int(total_height / len(train_dataset))
    print(f"Expected average image size: {w} x {h}")
    print()

    min_width, min_height, max_width, max_height = find_min_max_bbox_sizes(train_dataset)

    print(f"Recommended anchor sizes based on the dataset:")
    print(f"Minimum width x height: {min_width:.2f} x {min_height:.2f} pixels")
    print(f"Maximum width x height: {max_width:.2f} x {max_height:.2f} pixels")
    print()

    #---------------------------------------- MODEL INITIALIZATION ----------------------------------------#
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('CUDA available:', torch.cuda.is_available())
    print()

    pre_metadata = {
        'backbone_name'     : backbone_name,
        'num_classes'       : train_dataset.num_classes,
        'is_custom_anchor'  : is_custom_anchor,
        'anchor_sizes'      : anchor_sizes,
        'aspect_ratios'     : aspect_ratios,
        'model_version'     : model_version,
        'model_type'        : model_type,
        'trainable_layers'  : trainable_layers,
    }
    
    model = select_model(pre_metadata, device=device)

    model.to(device=device, dtype=torch.float32)

    model.device    = device
    model_name      = f'{model_type}_{backbone_name}_fpn_v{model_version}'

    #---------------------------------------- RESUME TRAINING ----------------------------------------#
    if resume:
        if model_path is not None:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            print('Model loaded for resume training!')
        else:
            print('No model path provided!')
        print()
    #---------------------------------------- TRAINING HYPERPARAMETERS ----------------------------------------#

    architecture_path = os.path.join(run_dir, "architecture.txt")
    with open(architecture_path, "w") as f:
        print(model, file=f)

    if optimizer_version == "simple":
        params          = [p for p in model.parameters() if p.requires_grad]

        optimizer       = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    elif optimizer_version == "grouped":
        decay, no_decay = set(), set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay.add(name)
            else:
                decay.add(name)

        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if n in decay], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if n in no_decay], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    if scheduler_type is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            anneal_strategy='cos'
        )

    class_colors    = generate_colors(len(train_dataset.class_names))

    metadata = Metadata(
        metadata_path   = metadata_path,
        training_data   = os.path.basename(dirroot),
        train_img_size  = (w, h),
        num_epochs      = num_epochs,
        early_stop      = early_stop,
        batch_size      = batch_size,
        num_workers     = num_workers,
        lr              = lr,
        weight_decay    = weight_decay,
        run_dir         = run_dir,
        num_classes     = train_dataset.num_classes,
        class_names     = train_dataset.class_names,
        class_colors    = {name: color for name, color in zip(train_dataset.class_names, class_colors)},
        model_type      = model_type,
        backbone_name   = backbone_name,
        model_version   = model_version,
        model_name      = model_name,
        trainable_layers= trainable_layers,
        is_custom_anchor= is_custom_anchor,
        is_resume       = resume,
        resume_model    = model_path,
    )

    metadata.add_attribute("anchor_sizes", anchor_sizes if is_custom_anchor else 'default')
    metadata.add_attribute("aspect_ratios", aspect_ratios if is_custom_anchor else 'default')

    metadata.save()
    print()

    #---------------------------------------- MODEL TRAINING ----------------------------------------#

    best_epoch                  = None
    metrics                     = {}
    best_metric                 = -float("inf")
    val_loss                    = float("inf")

    total_class_metrics         = {class_name: {"class_precision": 0, "class_recall": 0, "class_f1_score": 0, "class_accuracy": 0} for class_name in val_dataset.class_names if class_name != "none" and class_name != "background"}
    total_region_metrics        = {class_name: {"bbox_precision": 0, "bbox_recall": 0, "bbox_f1_score": 0, "mask_iou": 0} for class_name in val_dataset.class_names if class_name != "none" and class_name != "background"}

    early_stopping = EarlyStopping(patience=patience, delta=0.001, mode='max', path=run_dir)
    scaler = torch.amp.GradScaler()
    train_start = time.time()
    for epoch in range(num_epochs):
        epoch_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Learning Rate: {epoch_lr:.6f}")
        model.train()
        epoch_start = time.time()

        train_loss, train_classifier, train_box, train_mask = 0, 0, 0, 0
        train_batches = 0

        with tqdm(train_loader, desc="Training", ncols=100, leave=True) as pbar:
            for batch_idx, (imgs, targets) in enumerate(pbar):
                imgs    = [img.to(device, non_blocking=True) for img in imgs]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(imgs, targets) # loss_classifier, loss_box_reg, loss_mask (maskrcnn), loss_objectness, loss_rpn_box_reg
                    total_loss = sum(loss for loss in outputs.values()) 

                class_logits    = outputs.get('loss_classifier', 0.0)
                box_regression  = outputs.get('loss_box_reg', 0.0)
                mask_logits     = outputs.get('loss_mask', 0.0)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if scheduler_type is not None and batch_idx > 0:
                    scheduler.step()

                train_loss       += round(float(total_loss), 4)
                train_classifier += round(float(class_logits), 4)
                train_box        += round(float(box_regression), 4)
                train_mask       += round(float(mask_logits), 4)
                train_batches   += 1
        
        if train_batches > 0:
            train_loss      /= train_batches
            train_classifier/= train_batches
            train_box       /= train_batches
            train_mask      /= train_batches
        else:
            train_loss, train_classifier, train_box, train_mask = 0, 0, 0, 0
        
        print(
            f"Training Metrics - Total Loss: {train_loss:.4f}, "
            f"Loss Classifier: {train_classifier:.4f}, "
            f"Loss Box Reg: {train_box:.4f}"
            + (f", Loss Mask: {train_mask:.4f}" if model_type == "maskrcnn" else "")
            )

        val_loss, val_classifier, val_box, val_mask = 0, 0, 0, 0
        val_batches = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs    = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs     = model(imgs, targets)
                val_losses  = sum(loss for loss in outputs.values())
                
                class_logits    = outputs.get('loss_classifier', 0.0)
                box_regression  = outputs.get('loss_box_reg', 0.0)
                mask_logits     = outputs.get('loss_mask', 0.0)

                val_classifier += round(float(class_logits), 4)
                val_box        += round(float(box_regression), 4)
                val_mask       += round(float(mask_logits), 4)
                val_loss       += round(float(val_losses), 4)
                val_batches     += 1

        if val_batches > 0:
            val_classifier  /= val_batches
            val_box         /= val_batches
            val_mask        /= val_batches
            val_loss        /= val_batches
        else:
            val_loss, val_classifier, val_box, val_mask = 0, 0, 0, 0

    #---------------------------------------- MODEL VALIDATION ----------------------------------------#
        eval_results    = evaluate_model(epoch, model, val_dataset, val_loader, track_segment, debug_img_path, device, model_type=model_type)
        epoch_time      = time.time() - epoch_start

        overall         = eval_results["overall"]
        per_class_conf  = eval_results["per_class_confusion"]
        per_class_iou   = eval_results["per_class_iou"]
        iou_avg         = eval_results["iou_aggregated"]

        hrs, remainder  = divmod(epoch_time, 3600)
        mins, secs      = divmod(remainder, 60)
        epoch_time          = f"{int(hrs)}hrs {int(mins)}mins {secs:.2f}secs"

        print(
            f"Validation Metrics - Total Loss: {val_loss:.4f}, "
            f"Loss Classifier: {val_classifier:.4f}, Loss Box Reg: {val_box:.4f}"
            + (f", Loss Mask: {val_mask:.4f}" if model_type == "maskrcnn" else "")
        )

        print(
            f"Class Metrics - Precision: {overall['precision']:.4f}, "
            f"Recall: {overall['recall']:.4f}, F1-Score: {overall['f1_score']:.4f}, Accuracy: {overall['accuracy']:.4f}"
        )

        print(
            f"Bbox/ Mask Metrics - Precision: {iou_avg['avg_precision']:.4f}, "
            f"Recall: {iou_avg['avg_recall']:.4f}, F1-Score: {iou_avg['avg_f1_score']:.4f}"
            + (f", IoU: {iou_avg['avg_iou']:.4f}" if model_type == "maskrcnn" else "")
            + f", Time: {epoch_time}s"
        )

        if iou_avg['avg_f1_score'] > best_metric:
            best_metric = iou_avg['avg_f1_score']
            best_epoch  = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model at epoch {best_epoch} with BBox F1 Score: {best_metric:.4f}. Model saved.")

        torch.save(model.state_dict(), last_epoch_model_path)

        metrics = {
            'epoch': epoch + 1,
            'lr'                : epoch_lr,
            'train_loss'        : float(total_loss),
            'train_classifier'  : train_classifier,
            'train_box_reg'     : train_box,
            'train_mask'        : train_mask if model_type=='maskrcnn' else '-',
            'val_loss'          : val_loss,
            'val_classifier'    : val_classifier,
            'val_box_reg'       : val_box,
            'val_mask'          : val_mask if model_type=='maskrcnn' else '-',
            'val_precision'     : overall['precision'],
            'val_recall'        : overall['recall'],
            'val_f1_score'      : overall['f1_score'],
            'val_accuracy'      : overall['accuracy'],
            'time_taken'        : epoch_time,
        }

        for cls, metrics_dict in per_class_conf.items():
            if cls in {"none", "background"}:
                continue
            for metric_name, value in metrics_dict.items():
                if metric_name in {"TP", "FP", "FN"}:
                    continue
                key = f"{cls}:{metric_name}"
                total_class_metrics[cls][metric_name] += value
                metrics[key] = value


        for cls, metrics_dict in per_class_iou.items():
            if cls in {"none", "background"}:
                continue
            for metric_name, value in metrics_dict.items():
                key = f"{cls}:{metric_name}"
                total_region_metrics[cls][metric_name] += value
                metrics[key] = value

        try:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=metrics.keys())
                if epoch == 0:
                    writer.writeheader()
                writer.writerow(metrics)
        except IOError as e:
            print(f"Failed to write metrics to CSV: {e}")

        torch.cuda.empty_cache()
        early_stopping(iou_avg['avg_f1_score'], model, epoch + 1)

        if early_stop and early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        print("\n")

    #---------------------------------------- HISTORY LOG ----------------------------------------#
    averaged_total_class_metrics = {
        cls: {metric: total_class_metrics[cls][metric] / (epoch + 1) for metric in total_class_metrics[cls]}
        for cls in total_class_metrics
    }

    averaged_total_region_metrics = {
        cls: {metric: total_region_metrics[cls][metric] / (epoch + 1) for metric in total_region_metrics[cls]}
        for cls in total_region_metrics
    }

    train_time = time.time() - train_start
    create_segmentation_collages(model, val_dataset, device, class_colors, run_dir, best_model_path, model_type=model_type)
    plot_metrics(csv_file, run_dir)
    display_class_metrics_table(metrics_path, averaged_total_class_metrics, averaged_total_region_metrics, model_type=model_type)
    save_confusion_matrix(eval_results["per_class_confusion"], train_dataset.class_names, cm_path)

    print(f"Results up to epoch {epoch + 1} saved.")
    metadata.add_attribute("last_epoch", epoch + 1)
    metadata.save()

    hrs, remainder  = divmod(train_time, 3600)
    mins, secs      = divmod(remainder, 60)

    timing          = f"{int(hrs)}hrs {int(mins)}mins {secs:.2f}secs"

    print(f"\nTraining completed. Total time taken to complete training: {timing}.")
    if best_epoch == None:
        print("No best epoch to save")
    else:
        print(f"Best model saved at epoch {best_epoch}.")

if __name__ == '__main__':
    main(
    project_name        = "Project",
    dataset_location    = r"",
    result_path         = r'', # with "/results" folder ready
    resume              = False,
    model_path          = None,                                 # if want to resume model training
    num_workers         = 0,                                    # 'default':  max workers
    num_epochs          = 50,
    early_stop          = True,
    patience            = 10,                                   # 'default': round(num_epochs * 0.2)
    batch_size          = 4,
    lr                  = 0.0001,                               # fixed
    weight_decay        = 0.0001,                               # (wd) overfit increase, underfit decrease
    optimizer_version   = 'grouped',                            # simple(wd to all active params) or grouped(wd to only cov weights)
    scheduler_type      = 'oncylelr',
    track_segment         = {'save': False, 'limit' : 3},          # to debug 'mask generation' during evaluation phase
    model_type          = "maskrcnn",                           # "fasterrcnn: bbox" or "maskrcnn: instance segmentation"
    model_version       = '1',                                  # '1: custom' or '1.1: prepackaged' or '2: prepackaged'
    is_custom_anchor    = True,   
    # if is_custom_anchor == True
    anchor_sizes        = [(16,), (32,), (64,), (128,), (256,),],   # num of anchors follow num_feature_maps
    aspect_ratios       = [(0.5, 1.0, 2.0)],         # < 1: tall, > 1: wide  (follows num of anchor sizes)       
    # backbone_name only applicable for model_version == '1: custom'
    backbone_name       = "resnet34",                           # from torchvision.models import resnet
    trainable_layers    = 3,                                    # from 1 to 5
    roboflow_data       = None,                                 # roboflow_dataset().location or None
    )
