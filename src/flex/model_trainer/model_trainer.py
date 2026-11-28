from abc import ABC, abstractmethod
from typing import Any
import math
import torch
import torch.nn as nn

from .model_trainer_args import ModelTrainerArgs
from ..ml_utils.training_utils import TrainingUtils

class ModelTrainer(ABC):
    """
    Model trainer abstract base class
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        TrainingUtils.set_seed(42)
        self.trainer_args: ModelTrainerArgs = trainer_args

    # ------------------------------------------------------------------
    # Helper: transparently unwrap DataParallel before calling state_dict
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        """Strip nn.DataParallel wrapper so state_dict() keys are clean."""
        return model.module if isinstance(model, nn.DataParallel) else model

    @abstractmethod
    def train_step(self) -> float:
        """
        Performs a single training step.
        """
        pass

    @abstractmethod
    def train(self, epochs, is_return_wbab = False) -> Any:
        """
        Trains the model for a number of epochs.
        """
        pass

    def set_optimizer(self, optimizer):
        """
        Sets the optimizer for the trainer.
        """
        self.trainer_args.optimizer = optimizer

    def set_model(self, model):
        """
        Sets the model for the trainer.
        """
        self.trainer_args.model = model

    def set_train_loader(self, train_loader):
        """
        Sets the training data loader for the trainer.
        """
        self.trainer_args.train_loader = train_loader

    def _state_dict_l2_norm(self, state_dict: dict) -> float:
        """Compute L2 norm across all tensor values in a state_dict."""
        total_sq = 0.0
        for tensor in state_dict.values():
            if not isinstance(tensor, torch.Tensor):
                continue
            # Use float32 and move to CPU to avoid device-specific limitations (like MPS float64) and mismatched devices
            t = tensor.detach().to(device='cpu', dtype=torch.float32)
            total_sq += float(torch.sum(t * t).item())
        return math.sqrt(total_sq)

    def _state_dict_l2_distance(self, state_dict_a: dict, state_dict_b: dict) -> float:
        """Compute global Frobenius norm ||a-b||_F across all tensors in two state_dicts."""
        total_sq = 0.0
        common_keys = state_dict_a.keys() & state_dict_b.keys()
        for key in common_keys:
            ta = state_dict_a[key]
            tb = state_dict_b[key]
            if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
                continue
            # Move to CPU and use float32 for safe cross-device operations
            diff = ta.detach().to(device='cpu', dtype=torch.float32) - tb.detach().to(device='cpu', dtype=torch.float32)
            total_sq += float(torch.sum(diff * diff).item())
        return math.sqrt(total_sq)

    def _state_dict_l2_distance_layerwise(self, state_dict_a: dict, state_dict_b: dict) -> float:
        """Compute sum of per-layer L2 norms: Σ_l ||a_l - b_l||_2.

        Matches the original Keras implementation::

            total_norm = 0.0
            for w1, w2 in zip(weights1, weights2):
                diff = np.array(w1) - np.array(w2)
                total_norm += np.linalg.norm(diff)
            return total_norm

        Unlike the global Frobenius norm, this accumulates the L2 norm of each
        layer independently before summing, producing a larger value when
        divergence is spread across many layers.
        """
        total_norm = 0.0
        common_keys = sorted(state_dict_a.keys() & state_dict_b.keys())
        for key in common_keys:
            ta = state_dict_a[key]
            tb = state_dict_b[key]
            if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
                continue
            diff = ta.detach().to(device='cpu', dtype=torch.float32) - tb.detach().to(device='cpu', dtype=torch.float32)
            total_norm += float(torch.norm(diff, p=2).item())
        return total_norm

    def observe(self, epochs=5) -> Any:
        """
        Performs observation without updating the global state.
        """
        pass

    def extract_wbab(self):
        """
        Extracts structured model components (e.g., LoRA components).
        """
        pass
