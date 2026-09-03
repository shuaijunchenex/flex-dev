from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from ..ml_utils.key_value_args import KeyValueArgs

@dataclass
class ModelTrainerArgs(KeyValueArgs):
    """
    Trainer type
    """
    trainer_type: str = "standard"
    
    """
    Torch NN Model
    """
    model: nn.Module|None = None

    """
    Optimizer
    """
    optimizer: optim.Optimizer|None = None

    """
    Loss function
    """
    loss_func: Any = None

    """
    Training data
    """
    train_loader: DataLoader|None = None
    
    """
    Run on device via 'cpu' or 'gpu'
    """
    device: str = "cpu"

    """
    Result save folder
    """
    save_path: str = "./.trainer_result/"

    """
    """
    best_val_acc = 0.0

    def __init__(self, config_dict: dict|None = None, is_clone_dict = False):
        super().__init__(config_dict, is_clone_dict)

        if config_dict is not None and "trainer" in config_dict:
            self.set_args(config_dict["trainer"], is_clone_dict)

        self.dataset_type = self.get("trainer_type", "standard")
        self.device = self.get("device", "cpu")
        self.save_path = self.get("save_path", "./.trainer_result/")
        self.best_val_acc = self.get("best_val_acc", 0)
        
        return

    def set_trainer_args(self, model, optimizer, loss_func, train_loader, trainer_type="standard"):

        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.trainer_type = trainer_type

        return