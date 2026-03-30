from rfdetr.config import TrainConfig
from dataclasses import dataclass, field

# CUSTOM RFDETR TRAINING PARAMETERS
# for more parameters adjustment, kindly refer TrainConfig module at top :)
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-4
    progress_bar: str = "rich" # options: "rich" or "tqdm", None for disabled
    early_stopping: bool = True

# if CONFIG includes 'custom_model' (not available yet - in future)
# specfic model default configs are also within the TrainConfig script
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