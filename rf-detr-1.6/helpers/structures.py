# fcuntions to create containers to pass multiple elemetns across multi funcitons
from dataclasses import dataclass
from typing import Any, Dict, Type

@dataclass
class TrainingSession:
    model: Any
    train_cfg: Any
    dataset_dir: str
    session_dir: str
    dataset_info: Dict
    model_type: str
    model_class: Type