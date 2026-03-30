import os
import json
import torch
import platform
from datetime import datetime

# Class to save metadata of session training for future references
class TrainingMetadata:
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        experiment_name: str = "rf-detr-training",
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.created_at = datetime.now().isoformat()

        self.dataset = {}
        self.training = {}
        self.model = {}
        self.environment = self._collect_environment_info()
        self.results = {}

    def set_dataset_info(
        self,
        num_classes: int,
        class_names: list,
        colors: dict,
        annotation_path: str,
        train_split: str = "train",
        valid_split: str = "valid",
        test_split: str = "test",
    ):
        self.dataset = {
            "dataset_dir": self.dataset_dir,
            "output_dir": self.output_dir,
            "annotation_path": annotation_path,
            "splits": {
                "train": train_split,
                "valid": valid_split,
                "test": test_split,
            },
            "num_classes": num_classes,
            "class_names": class_names,
            "colors": colors
        }

    def set_training_params(self, **kwargs):
        self.training = kwargs

    def set_model_config(self, **kwargs):
        self.model = kwargs

    def set_results(self, **kwargs):
        """
        For metrics, checkpoints, inference stats, etc.
        """
        self.results = kwargs

    def to_dict(self):
        return {
            "created_at": self.created_at,
            "dataset": self.dataset,
            "training": self.training,
            "model": self.model,
            "environment": self.environment,
            "results": self.results,
        }

    def save(self, filename: str = "training_metadata.json"):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        return path

    @staticmethod
    def load(path: str):
        with open(path, "r") as f:
            return json.load(f)

    def _collect_environment_info(self):
        env = {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            env["gpu"] = {
                "name": props.name,
                "total_vram_gb": round(props.total_memory / 1024**3, 2),
                "cuda_version": torch.version.cuda,
            }

        return env
