from dataclasses import dataclass

@dataclass
class ModelConfig:
    encoder: str = "dinov2_windowed_small"
    resolution: int = 432
    num_queries: int = 300
    num_select: int = 300
    patch_size: int = 12
    dec_layers: int = 4
    num_windows: int = 2
    segmentation_head: bool = True
    # mask_point_sample_ratio: int = 8 #default 16
    # mask_downsample_ratio: int = 4 #default 4


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 2
    grad_accum_steps: int = 4
    lr: float = 1e-4
    lr_encoder: float = 1e-5
    early_stopping: bool = True
    early_stopping_patience: int = 10
    num_workers: int = 0
    multi_scale: bool = True
    expanded_scales: bool = False
    use_ema: bool = True
    # group_detr: int = 4 #default 13
