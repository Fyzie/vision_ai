import time
import random
import colorsys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np
import datetime
import os
import glob
from matplotlib import pyplot as plt

from tqdm import tqdm
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage
from utils import timer
from modules.build_yolact import Yolact
from modules.multi_loss import Multi_Loss
from data.config import cfg, update_config
from data.coco import COCODetection, detection_collate
from eval import evaluate
from annotators import mask_annotator, box_annotator, label_annotator

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

def create_segmentation_collages(model, val_loader, device, class_colors, run_dir, best_model_path, sample_ratio=1.0, threshold_overlay=0.5):
    """
    Create image collages of actual vs predicted masks and boxes.
    """
    import random
    from utils.output_utils import after_nms, NMS
    from utils.augmentations import BaseTransform

    print('[NOTE] Creating segmentation collages\n')

    model.eval()
    model.load_weights(best_model_path, device.type == 'cuda')
    model.to(device)

    transform = BaseTransform()
    sample_size = max(1, int(len(val_loader.dataset) * sample_ratio))
    sampled_indices = random.sample(range(len(val_loader.dataset)), sample_size)

    actual_images = []
    predicted_images = []

    for idx in sampled_indices:
        print("Val Loader")
        print(val_loader)
        datum = val_loader.dataset[idx]
        img_tensor, target = data_to_device(datum, device.type == 'cuda', is_eval=True)

        # --- Normalize image shape ---
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor).float()

        if img_tensor.ndim == 2:  # grayscale H,W
            img_tensor = img_tensor.unsqueeze(0)  # -> (1,H,W)
        elif img_tensor.ndim == 3 and img_tensor.shape[0] not in [1, 3]:
            # probably HWC -> convert to CHW
            img_tensor = img_tensor.permute(2, 0, 1)

        # Add batch dimension for model
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1,C,H,W)

        # --- Make a numpy copy for visualization ---
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        with torch.no_grad():
            preds_raw = model(img_tensor)
            preds_nms = NMS(preds_raw, traditional_nms=False)
            results = after_nms(preds_nms, img_np.shape[0], img_np.shape[1], visual_thre=threshold_overlay)

        # ---------- Ground Truth ----------
        actual_boxes = target[0].cpu().numpy() if isinstance(target, (list, tuple)) else target['labels'][0].cpu().numpy()
        actual_masks = target[1] if isinstance(target, (list, tuple)) else target['masks'][0]
        actual_labels = [cfg.dataset.class_names[label.item()] for label in target[2]] if isinstance(target, (list, tuple)) else []
        actual_colors = [class_colors[label] for label in target[2]] if isinstance(target, (list, tuple)) else []

        img_actual = mask_annotator(img_np, actual_masks, actual_colors, alpha=0.5)
        img_actual = box_annotator(img_actual, actual_boxes, actual_colors)
        img_actual = label_annotator(img_actual, actual_boxes, actual_labels, actual_colors)

        # ---------- Prediction Visualization ----------
        pred_boxes = results['boxes']
        pred_scores = results['scores']
        pred_labels = results['classes']
        pred_masks = results['masks']

        pred_colors = [class_colors[label] for label in pred_labels]
        pred_labels_str = [f"{cfg.dataset.class_names[label]}: {score:.2f}" for label, score in zip(pred_labels, pred_scores)]

        img_pred = mask_annotator(img_np, pred_masks, pred_colors, alpha=0.5)
        img_pred = box_annotator(img_pred, pred_boxes, pred_colors)
        img_pred = label_annotator(img_pred, pred_boxes, pred_labels_str, pred_colors)

        # Store for collage
        actual_images.append(torch.from_numpy(img_actual).permute(2, 0, 1) / 255.0)
        predicted_images.append(torch.from_numpy(img_pred).permute(2, 0, 1) / 255.0)

    # --- Save collages ---
    def save_collage(images, title, filename, cell_size=5, label_size=12):
        cols = 3
        rows = (len(images) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_size, rows * cell_size))
        for ax, img in zip(axes.flatten(), images):
            ax.imshow(img.permute(1, 2, 0).cpu())
            ax.axis("off")
        for ax in axes.flatten()[len(images):]:
            ax.axis("off")
        plt.suptitle(title, fontsize=label_size * 2)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        os.makedirs(run_dir, exist_ok=True)
        plt.savefig(os.path.join(run_dir, filename))
        plt.close()

    for i in range(0, len(actual_images), 9):
        save_collage(actual_images[i:i+9], "Ground Truth", f"{i//9+1}_actual.png")
        save_collage(predicted_images[i:i+9], "Predictions", f"{i//9+1}_predicted.png")

    print(f"[DONE] Saved segmentation collages to {run_dir}")


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def data_to_device(datum, cuda, is_eval=False):
    images, boxes, masks, num_crowds = datum

    if is_eval:
        images = to_tensor(images).float()
        boxes  = [to_tensor(b) for b in boxes]
        masks  = [to_tensor(m) for m in masks]

    if cuda:
        images = images.cuda().detach()
        boxes = [ann.cuda().detach() for ann in boxes]
        masks = [mask.cuda().detach() for mask in masks]
    else:
        images = images.detach()
        boxes = [ann.detach() for ann in boxes]
        masks = [mask.detach() for mask in masks]

    targets = {
        'labels': boxes,
        'masks': masks,
        'crowds': num_crowds,
    }

    return images, targets

def compute_val_map(yolact_net, val_dataset):
    with torch.no_grad():
        yolact_net.eval()
        print("\nComputing validation mAP...", flush=True)
        global_table, class_table, box_row, mask_row = evaluate(yolact_net, val_dataset, during_training=True)
        yolact_net.train()
        return global_table, class_table, box_row[1], mask_row[1]


def print_result(map_tables):
    print('\nValidation results during training:\n')
    for info, table in map_tables:
        print(info)
        print(table, '\n')


def save_best(net, mask_map, step, weights_path, cuda, best_model_path=None):
    weight = glob.glob(f'{weights_path}/best*')
    best_mask_map = float(weight[0].split('/')[-1].split('_')[1]) if weight else 0.
    if mask_map >= best_mask_map:
        if weight:
            os.remove(weight[0])  # remove the last best model
        print(f'\nSaving the current best model as \'best_{mask_map}_{cfg.name}_{step}.pth\'.\n')
        best_model_path = f'{weights_path}/best_{mask_map}_{cfg.name}_{step}.pth'
        torch.save(net.state_dict(), best_model_path)

    return best_model_path


def save_latest(net, step, weights_path):
    weight = glob.glob(f'{weights_path}/latest*')
    if weight:
        os.remove(weight[0])
    torch.save(net.state_dict(), f'{weights_path}/latest_{cfg.name}_{step}.pth')

def main(
        project_name= "Project",
        dataset_path     = None,
        result_path = None,
        pretrained_path = 'weights',
        config      = "res101_custom_config",
        num_epochs  = 5,
        batch_size  = 8,
        lr          = 0.0001,
        weight_decay= 0.0001,
        img_size    = 550,
        resume      = None,
        model_type  = "yolact",
        model_version = 3,

        ):
    
    script_dir      = os.path.dirname(os.path.abspath(__file__))
    if result_path == None:
        result_path = os.path.join(script_dir, "results", project_name)
    else:
        result_path = os.path.join(result_path, "results", project_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Created directory: {result_path}")
    os.chdir(result_path)

    run_dir = f"{result_path}/run{len([d for d in os.listdir(result_path) if d.startswith('run')]) + 1}"
    os.makedirs(run_dir, exist_ok=True)

    weights_path    = os.path.join(run_dir, "weights")
    metadata_path   = os.path.join(run_dir, "metadata.json")


    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    train_img_size = max(img_size)
    update_config(config, dataset_path, batch_size, lr, weight_decay, train_img_size)
    print('\n' + '-' * 30 + 'Configs' + '-' * 30)
    for k, v in vars(cfg).items():
        print(f'{k}: {v}')

    timer.disable_all()

    cuda = torch.cuda.is_available()
    torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = Multi_Loss(num_classes=cfg.num_classes, pos_thre=cfg.pos_iou_thre, neg_thre=cfg.neg_iou_thre, np_ratio=3)
    net = Yolact(criterion=criterion)

    if resume == 'latest':
        weight = glob.glob(f'{pretrained_path}/latest*')[0]
        net.load_weights(weight, cuda)
        resume_step = int(weight.split('.pth')[0].split('_')[-1])
        print(f'\nResume training with \'{weight}\'.\n')
    elif resume and 'yolact' in resume:
        net.load_weights(f'{pretrained_path}/' + resume, cuda)
        resume_step = int(resume.split('.pth')[0].split('_')[-1])
        print(f'\nResume training with \'{resume}\'.\n')
    else:
        backbone_path = os.path.join(pretrained_path, cfg.backbone.path)
        print(f"Bacbbone Path: {backbone_path}")
        net.init_weights(backbone_path=backbone_path)
        resume_step = 0
        print('\nTraining from beginning, weights initialized.\n')

    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.decay)

    if cuda:
        cudnn.benchmark = True
        net = net.to(device)
        criterion = criterion.to(device)

    dataset = COCODetection(image_path=cfg.dataset.train_images, info_file=cfg.dataset.train_info,
                            augmentation=SSDAugmentation())

    data_loader = data.DataLoader(dataset, cfg.batch_size, num_workers=2, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)
    
    val_dataset = COCODetection(
        image_path=cfg.dataset.valid_images,
        info_file=cfg.dataset.valid_info,
        augmentation=BaseTransform()
    )

    val_loader = data.DataLoader(val_dataset, cfg.batch_size, num_workers=2, shuffle=False,
                                collate_fn=detection_collate, pin_memory=True)

    iters_per_epoch = len(data_loader)
    cfg.max_iter = iters_per_epoch * num_epochs
    print(f"Training for {cfg.max_iter} iterations (~{num_epochs} epochs).")

    class_names     = cfg.dataset.class_names
    class_colors    = generate_colors(len(class_names))

    metadata = Metadata(
        metadata_path   = metadata_path,
        training_data   = dataset_path,
        train_img_size  = (img_size[0],img_size[1]),
        num_epochs      = num_epochs,
        batch_size      = batch_size,
        lr              = lr,
        weight_decay    = weight_decay,
        run_dir         = run_dir,
        num_classes     = cfg.num_classes,
        class_names     = class_names,
        class_colors    = {name: color for name, color in zip(class_names, class_colors)},
        model_type      = model_type,
        backbone_name   = config,
        model_version   = model_version,
        is_resume       = resume,
    )

    metadata.save()

    step_index = 0
    start_step = resume_step
    batch_time = MovingAverage()
    loss_types = ['loss_box_reg', 'loss_classifier', 'loss_mask', 'loss_semantic']
    loss_avgs = {k: MovingAverage() for k in loss_types}
    map_tables = []
    best_model_path = None

    try:
        for epoch in range(num_epochs):
            epoch_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch [{epoch + 1}/{num_epochs}] - Learning Rate: {epoch_lr:.6f}")
            net.train()

            for k in loss_types:   # reset moving averages per epoch
                loss_avgs[k].reset()

            with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, leave=False) as pbar:
                for i, datum in enumerate(pbar):
                    step = epoch * iters_per_epoch + i + start_step

                    # Warmup LR
                    if cfg.warmup_until > 0 and step <= cfg.warmup_until:
                        set_lr(optimizer, (cfg.lr - cfg.warmup_init) * (step / cfg.warmup_until) + cfg.warmup_init)

                    # LR schedule
                    while step_index < len(cfg.lr_steps) and step >= cfg.lr_steps[step_index]:
                        step_index += 1
                        set_lr(optimizer, cfg.lr * (0.1 ** step_index))

                    images, targets = data_to_device(datum, cuda)

                    if cuda: torch.cuda.synchronize()
                    predictions, outputs = net(images, targets=targets)

                    total_loss = sum(loss for loss in outputs.values())

                    optimizer.zero_grad()
                    total_loss.backward()
                    if torch.isfinite(total_loss).item():
                        optimizer.step()

                    for k in outputs:
                        loss_avgs[k].add(outputs[k].item())

            avg_losses = {k: loss_avgs[k].get_avg() for k in loss_types}
            train_loss = sum(avg_losses.values())

            print(
                f"Training Metrics - Total Loss: {train_loss:.4f}, "
                f"Loss Classifier: {avg_losses.get('loss_classifier', 0):.4f}, "
                f"Loss Box Reg: {avg_losses.get('loss_box_reg', 0):.4f}, "
                f"Loss Mask: {avg_losses.get('loss_mask', 0):.4f}"
            )

            table, class_table, box_map, mask_map = compute_val_map(net, val_dataset)

            info = (f'Epoch {epoch+1} | step {step}')
            map_tables.append((info, table))
            save_latest(net, step, weights_path)
            best_model_path = save_best(net, mask_map, step, weights_path, cuda, best_model_path=best_model_path)

    except KeyboardInterrupt:
        print(f'\nStopped, saving the latest model as \'latest_{cfg.name}_{step}.pth\'.\n')
        save_latest(net, step, weights_path)
        print_result(map_tables)
        exit()

    print(f'Training completed, saving the final model as \'latest_{cfg.name}_{step}.pth\'.\n')
    save_latest(net, step, weights_path)
    print_result(map_tables)
    create_segmentation_collages(
        model           = net,
        val_loader     = val_loader,
        device          = device,
        class_colors    = {i: c for i, c in enumerate(class_colors)},
        run_dir         = run_dir,
        best_model_path = best_model_path,
        sample_ratio    = 0.1,         # 10% of val set
        threshold_overlay = 0.3
    )


if __name__ == "__main__":
    main(
        project_name    = "LeadFrame",
        dataset_path    = r"D:\Pytorch Projects\computer_vision\training\data\lead_frame-8-augment.1",
        result_path     = r'D:\Pytorch Projects\computer_vision\training',
        pretrained_path = r'D:\Pytorch Projects\computer_vision\training\weights',
        config          = "res101_custom_config",
        num_epochs      = 1,
        batch_size      = 8,
        lr              = 0.001,
        weight_decay    = 0.0001,
        img_size        = [640, 640],
        resume          = None,
        )
