import os
import json
import numpy as np
from modules.backbone import ResNetBackbone

COLORS = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')

LABEL_MAP = {i: i for i in range(len(COLORS))}

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)

class Config(object):
    """
    After implement this class, you can call 'cfg.x' instead of 'cfg['x']' to get a certain parameter.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making the changes given by new_config_dict.
        """
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object. Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def __repr__(self):
        return self.name

# ----------------------- TRANSFORMS ----------------------- #
resnet_transform = Config({'channel_order': 'RGB',
                           'normalize': True,
                           'subtract_means': False,
                           'to_float': False})

# ----------------------- BACKBONES ----------------------- #
resnet101_backbone = Config({'name': 'ResNet101',
                             'path': 'resnet101_reducedfc.pth',
                             'type': ResNetBackbone,
                             'args': ([3, 4, 23, 3],),
                             'transform': resnet_transform,
                             'selected_layers': [1, 2, 3]})

resnet50_backbone = resnet101_backbone.copy({'name': 'ResNet50',
                                             'path': 'resnet50-19c8e357.pth',
                                             'args': ([3, 4, 6, 3],)})

res101_coco_config = Config({
    'name': 'res101_coco',
    'dataset': None,
    'num_classes': None,
    'batch_size': 8,
    'img_size': 550,  # image size
    'max_iter': 800000,
    'backbone': resnet101_backbone,
    # During training, first compute the maximum gt IoU for each prior.
    # Then, for priors whose maximum IoU is over the positive threshold, marked as positive.
    # For priors whose maximum IoU is less than the negative threshold, marked as negative.
    # The rest are neutral ones and are not used to calculate the loss.
    'pos_iou_thre': 0.5,
    'neg_iou_thre': 0.4,
    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 0.7,
    'conf_alpha': 1,
    'bbox_alpha': 1.5,
    'mask_alpha': 6.125,
    # Learning rate
    'lr_steps': (280000, 600000, 700000, 750000),
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,
    # warm up setting
    'warmup_init': 1e-4,
    'warmup_until': 500,
    # The max number of masks to train for one image.
    'masks_to_train': 100,
    # anchor settings
    'scales': [24, 48, 96, 192, 384],
    'aspect_ratios': [1, 1 / 2, 2],
    'use_square_anchors': True,  # This is for backward compatability with a bug.
    # Whether to train the semantic segmentations branch, this branch is only implemented during training.
    'train_semantic': True,
    'semantic_alpha': 1,
    # postprocess hyperparameters
    'conf_thre': 0.05,
    'nms_thre': 0.5,
    'top_k': 200,
    'max_detections': 100,
    # Freeze the backbone bn layer during training, other additional bn layers after the backbone will not be frozen.
    'freeze_bn': False,
    'label_map': None})

mask_proto_net = [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}),
                  (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})]

extra_head_net = [(256, 3, {'padding': 1})]

res50_coco_config = res101_coco_config.copy({'name': 'res50_coco','backbone': resnet50_backbone})

res101_custom_config = res101_coco_config.copy({'name': 'res101_custom'})

res50_custom_config = res50_coco_config.copy({'name': 'res50_custom'})

def update_config(config, dataset=None, batch_size=None, lr=None, weight_decay=None, img_size=None):
    global cfg
    cfg.replace(eval(config))

    if dataset:
        train_images    = os.path.join(dataset, "train")
        train_info      = os.path.join(dataset, "train/_annotations.coco.json")
        valid_images    = os.path.join(dataset, "valid")
        valid_info      = os.path.join(dataset, "valid/_annotations.coco.json")

        with open(train_info, "r") as f:
            train_json = json.load(f)

        CUSTOM_CLASSES = tuple([cat["name"] if idx != 0 else "background" for idx, cat in enumerate(train_json["categories"])])

        CUSTOM_LABEL_MAP = {i: i for i in range(len(CUSTOM_CLASSES))}

        num_classes = len(CUSTOM_CLASSES) + 1

        custom_dataset = Config({'name': 'Custom dataset',
                         'train_images': train_images,
                         'train_info': train_info,
                         'valid_images': valid_images,
                         'valid_info': valid_info,
                         'class_names': CUSTOM_CLASSES})
        
        setattr(cfg, 'dataset', custom_dataset)
        setattr(cfg, 'label_map', CUSTOM_LABEL_MAP)
        setattr(cfg, 'num_classes', num_classes)

    if batch_size:
        setattr(cfg, 'batch_size', batch_size)
    if lr:
        setattr(cfg, 'lr', lr)
    if weight_decay:
        setattr(cfg, 'weight_decay', weight_decay)
    if img_size:
        setattr(cfg, 'img_size', img_size)
        scales = [int(img_size / 550 * aa) for aa in cfg.scales]
        setattr(cfg, 'scales', scales)

cfg = res101_coco_config.copy()