import cv2
import json

import torch
import torch.nn.functional as torch_F

import torchvision
import torchvision.ops as ops
from torchvision import models as mdl
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator


def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def select_model(metadata, device='cuda'):
    backbone_name       = metadata.get('backbone_name')
    num_classes         = metadata.get('num_classes')
    is_custom_anchor    = metadata.get('is_custom_anchor')
    anchor_sizes        = metadata.get('anchor_sizes')
    aspect_ratios       = metadata.get('aspect_ratios')
    model_version       = metadata.get('model_version')
    model_type          = metadata.get('model_type')
    trainable_layers    = metadata.get('trainable_layers')

    resnet_weights_map = {
        "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
        "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
        "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
        "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
        "resnet152": torchvision.models.ResNet152_Weights.DEFAULT,
    }

    backbone_weights = resnet_weights_map.get(backbone_name, None)


    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    backbone        = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name = backbone_name, weights=backbone_weights, trainable_layers=trainable_layers)
    backbone.to(device)

    dummy_image         = torch.randn(1, 3, 512, 512).to(device)

    features            = backbone(dummy_image)

    num_feature_maps    = len(features)

    for level, feature_map in features.items():
        print(f"Feature Map {level}: Shape {feature_map.shape}")
    print()

    #----------------------------------
    # custom anchor generator
    #----------------------------------
    if is_custom_anchor:
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    else:
        from torchvision.models.detection.faster_rcnn import _default_anchorgen
        anchor_generator = _default_anchorgen()
    #----------------------------------

    if model_version == '1':
        if model_type == "maskrcnn":
            print("[MODEL TYPE] Mask R-CNN V1 (custom)")
            model = MaskRCNN(
                backbone            = backbone,
                num_classes         = num_classes,
                rpn_anchor_generator= anchor_generator
            )
            
        elif model_type == "fasterrcnn":
            print("[MODEL TYPE] Faster R-CNN V1 (custom)")
            model = FasterRCNN(
                backbone            = backbone,
                num_classes         = num_classes,
                rpn_anchor_generator= anchor_generator,
            )
        
    else:
        backbone_name = "resnet50"
        if model_version == '1.1':
            if model_type == "maskrcnn":
                print("[MODEL TYPE] Mask R-CNN V1 (prepackaged)")
                model = maskrcnn_resnet50_fpn(
                    weights                     = MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                    weights_backbone            = mdl.ResNet50_Weights.DEFAULT,
                    trainable_backbone_layers   = trainable_layers,
                )

            elif model_type == "fasterrcnn":
                print("[MODEL TYPE] Faster R-CNN V1 (prepackaged)")
                model = fasterrcnn_resnet50_fpn(
                    weights                     = FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                    weights_backbone            = mdl.ResNet50_Weights.DEFAULT,
                    trainable_backbone_layers   = trainable_layers,
                )
        
        elif model_version == '2':
            if model_type == "maskrcnn":
                print("[MODEL TYPE] Mask R-CNN V2 (prepackaged)")
                model = maskrcnn_resnet50_fpn_v2(
                    weights                     = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                    weights_backbone            = mdl.ResNet50_Weights.DEFAULT,
                    trainable_backbone_layers   = trainable_layers,
                )

            elif model_type == "fasterrcnn":
                print("[MODEL TYPE] Faster R-CNN V2 (prepackaged)")
                model = fasterrcnn_resnet50_fpn_v2(
                    weights                     = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                    weights_backbone            = mdl.ResNet50_Weights.DEFAULT,
                    trainable_backbone_layers   = trainable_layers,
                )
            
        model.rpn.anchor_generator = anchor_generator

    print()
    print("Anchor Sizes     :", anchor_generator.sizes)
    print("Aspect Ratios    :", anchor_generator.aspect_ratios)

    in_features_box         = model.roi_heads.box_predictor.cls_score.in_features
    if model_type == "maskrcnn":
        in_features_mask    = model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced         = model.roi_heads.mask_predictor.conv5_mask.out_channels

    model.roi_heads.box_predictor       = FastRCNNPredictor(in_channels=in_features_box, num_classes=num_classes)
    if model_type == "maskrcnn":
        model.roi_heads.mask_predictor  = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=num_classes)

    return model


def load_model(model_path, metadata, half=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('CUDA available:', torch.cuda.is_available())
    print()

    model = select_model(metadata, device=device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    
    if half:
        model.half()
        dtype=torch.float16
    else:
        dtype=torch.float32

    model.to(device=device, dtype=dtype).eval()
    return model, half

def pred(image, models, metadata, conf_thresh=0.5, nms_threshold = 0.5, with_score=True):
    device      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, half = models
    model_type  = metadata.get('model_type')
    class_colors= metadata.get('class_colors')
    class_names = metadata.get('class_names')
    img_sz      = metadata.get('train_img_size')

    predicted_image = cv2.resize(image, img_sz)
    with torch.no_grad():
        input_tensor = F.to_tensor(predicted_image).unsqueeze(0).to(device)
        if half:
            input_tensor = input_tensor.half()
        output = model(input_tensor)[0]

        high_conf_indices   = output['scores'] > conf_thresh
        scores    = output['scores'][high_conf_indices]
        masks     = output['masks'][high_conf_indices] if model_type == 'maskrcnn' else None
        boxes     = output['boxes'][high_conf_indices]
        labels    = output['labels'][high_conf_indices]

        keep = apply_nms(boxes, scores, labels, iou_threshold=nms_threshold, is_unique=False)
        predicted_boxes = boxes[keep].cpu().numpy()
        predicted_scores = scores[keep].cpu()
        predicted_labels = labels[keep]
        predicted_masks = masks[keep] if masks is not None else None

        predicted_colors    = [class_colors[class_names[label.item()]] for label in predicted_labels]
        predicted_labels = [
            f"{class_names[label.item()]}: {score:.2f}" if with_score else class_names[label.item()]
            for label, score in zip(predicted_labels, predicted_scores)
        ]

        def resize_and_binarize_masks(masks, image_size):
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            resized_masks   = torch_F.interpolate(masks, size=image_size, mode="bilinear", align_corners=False)
            binarized_masks = (resized_masks > 0.5).byte()
            return binarized_masks.squeeze(1)
        
        if predicted_masks is not None:
            predicted_masks = resize_and_binarize_masks(predicted_masks, predicted_image.shape[:2])
        else:
            predicted_masks = None
        
    return predicted_image, predicted_scores, predicted_boxes, predicted_masks, predicted_labels, predicted_colors


def apply_nms(boxes, scores, labels, iou_threshold=0.2, is_unique=True):
    """
    Apply Non-Maximum Suppression (NMS) per class.

    Args:
        boxes (Tensor[N, 4]): Bounding boxes
        scores (Tensor[N]): Confidence scores
        labels (Tensor[N]): Class labels
        iou_threshold (float): IoU threshold for suppression

    Returns:
        keep_indices (Tensor): Indices to keep
    """
    if is_unique:
        keep = []
        unique_labels = labels.unique()
        for cls in unique_labels:
            cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
            cls_boxes = boxes[cls_indices]
            cls_scores = scores[cls_indices]
            keep_indices = ops.nms(cls_boxes, cls_scores, iou_threshold)
            keep.extend(cls_indices[keep_indices].tolist())
        return torch.tensor(keep, dtype=torch.long)
    else:
        keep = ops.nms(boxes, scores, iou_threshold)
        return keep

