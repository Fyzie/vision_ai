# -*- coding: utf-8 -*-
import torch
from itertools import product
from math import sqrt
from data.config import cfg


def center_size(boxes):
    """ Convert prior_boxes to format: (cx, cy, w, h)."""
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)


def intersect(box_a, box_b):
    """
    Compute intersection areas between two sets of boxes.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n, A, 4] or [A,4]
      box_b: (tensor) bounding boxes, Shape: [n, B, 4] or [B,4]
    Return:
      (tensor) intersection area, Shape: [n,A,B] or [A,B]
    """
    # Ensure both tensors are on the same device and type
    if box_a.device != box_b.device:
        box_b = box_b.to(box_a.device)
    # Ensure float dtype for calculations
    if not torch.is_floating_point(box_a):
        box_a = box_a.float()
    if not torch.is_floating_point(box_b):
        box_b = box_b.float()

    # Handle non-batched input by converting to batch dim
    squeeze = False
    if box_a.dim() == 2 and box_b.dim() == 2:
        squeeze = True
        box_a = box_a.unsqueeze(0)
        box_b = box_b.unsqueeze(0)

    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)

    max_xy = torch.min(
        box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0.0)

    out = inter[:, :, :, 0] * inter[:, :, :, 1]
    return out.squeeze(0) if squeeze else out


def jaccard(box_a, box_b, iscrowd: bool = False):
    """
    Compute the IoU (Jaccard) of two sets of boxes.
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4] or [B, num_objects, 4]
        box_b: (tensor) Prior boxes or predicted boxes, Shape: [num_priors,4] or [B, num_priors, 4]
        iscrowd: if True, compute intersection / area_a (COCO crowd behavior)
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)] or [B, A, B]
    """
    # Align devices and dtypes
    if box_a.device != box_b.device:
        box_b = box_b.to(box_a.device)
    if not torch.is_floating_point(box_a):
        box_a = box_a.float()
    if not torch.is_floating_point(box_b):
        box_b = box_b.float()

    use_batch = True
    if box_a.dim() == 2 and box_b.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)  # [B, A, C]

    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)

    union = area_a + area_b - inter
    # Avoid division by zero
    eps = 1e-7
    out = inter / (area_a + eps) if iscrowd else inter / (union + eps)

    return out if use_batch else out.squeeze(0)


def match(pos_thresh, neg_thresh, box_gt, priors, class_gt, crowd_boxes):
    """
    Match each prior box with the ground truth box of the highest jaccard overlap, encode the bounding boxes,
    then return the matched indices corresponding to both confidence and location preds.
    """
    # Work with tensors on the same device
    device = priors.device
    if box_gt.device != device:
        box_gt = box_gt.to(device)
    if class_gt.device != device:
        class_gt = class_gt.to(device)

    # Convert prior boxes to the form of [xmin, ymin, xmax, ymax].
    priors_detached = priors.detach()
    decoded_priors = torch.cat(
        (priors_detached[:, :2] - priors_detached[:, 2:] / 2, priors_detached[:, :2] + priors_detached[:, 2:] / 2),
        1,
    )

    overlaps = jaccard(box_gt, decoded_priors)  # size: [num_objects, num_priors]

    each_box_max, each_box_index = overlaps.max(1)  # size [num_objects]
    each_prior_max, each_prior_index = overlaps.max(0)  # size [num_priors]

    # Guarantee at least one prior matches each gt box
    each_prior_max.index_fill_(0, each_box_index, 2.0)

    # Set the index of the pair (prior, gt) we set the overlap for above.
    for j in range(each_box_index.size(0)):
        each_prior_index[each_box_index[j]] = j

    each_prior_box = box_gt[each_prior_index]  # size: [num_priors, 4]
    conf = class_gt[each_prior_index].long() + 1  # the class of the max IoU gt box for each prior

    conf[each_prior_max < pos_thresh] = -1  # label as neutral
    conf[each_prior_max < neg_thresh] = 0  # label as background

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)  # [num_priors, num_crowds]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1

    offsets = encode(each_prior_box, priors_detached)

    return offsets, conf, each_prior_box, each_prior_index


def make_anchors(conv_h, conv_w, scale):
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        # + 0.5 because priors are in center
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in cfg.aspect_ratios:
            ar_sqrt = sqrt(ar)
            w = scale * ar_sqrt / cfg.img_size
            h = scale / ar_sqrt / cfg.img_size

            # Backwards compat: square anchors option
            if cfg.use_square_anchors:  # True/False
                h = w

            prior_data += [x, y, w, h]

    return prior_data


def encode(matched, priors):
    variances = [0.1, 0.2]

    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    offsets = torch.cat([g_cxcy, g_wh], 1)

    return offsets


def decode(box_p, priors):
    """
    Decode predicted bbox coordinates
    """
    variances = [0.1, 0.2]

    # Make sure priors are on the same device as predictions
    priors = priors.to(box_p.device)

    boxes = torch.cat(
        (
            priors[:, :2] + box_p[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(box_p[:, 2:] * variances[1]),
        ),
        1,
    )

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def sanitize_coordinates(_x1, _x2, img_size: int, padding: int = 0):
    """
    Sanitizes input coordinates; converts from relative to absolute coordinates and clamps to image bounds.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    masks: [h, w, n]
    boxes: [n, 4] in relative point form (xmin, ymin, xmax, ymax)
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


def mask_iou(mask1, mask2, iscrowd=False):
    """
    Inputs are matrices of size N x M (bool/float). Output is size N x M.
    """
    # Align both tensors to the same device & dtype
    if mask1.device != mask2.device:
        mask2 = mask2.to(mask1.device)
    if not torch.is_floating_point(mask1):
        mask1 = mask1.float()
    if not torch.is_floating_point(mask2):
        mask2 = mask2.float()

    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    eps = 1e-7
    if iscrowd:
        ret = intersection / (area1.t() + eps)
    else:
        ret = intersection / (union + eps)

    return ret


def bbox_iou(bbox1, bbox2, iscrowd=False):
    """
    bbox1: [N, 4], bbox2: [M, 4] (both in absolute pixel coords or same scale)
    Output: IoU matrix [N, M]
    """
    # Align devices and dtypes
    if bbox1.device != bbox2.device:
        bbox2 = bbox2.to(bbox1.device)
    if not torch.is_floating_point(bbox1):
        bbox1 = bbox1.float()
    if not torch.is_floating_point(bbox2):
        bbox2 = bbox2.float()

    return jaccard(bbox1, bbox2, iscrowd)
