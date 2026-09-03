from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
import torch.nn as nn
from torch.optim import Optimizer
from ..ml_algorithms import OptimizerBuilder
from .strategy_args import StrategyArgs
from .base_strategy import BaseStrategy
from ..ml_utils.model_utils import ModelUtils
from ..ml_utils.gpu_memory_cleaner import GPUMemoryCleaner
from ..ml_utils import console


class ClientStrategy(BaseStrategy):
    """Abstract base for a client's local-training/observation strategy."""

    def __init__(self) -> None:
        super().__init__()
        self._strategy_type : str = "client"
        self._obj = None
        self._auto_offload: bool = True  # per‑strategy offload override

    def create(self, args: StrategyArgs, client_node):
        self._args = args
        self._create_inner(args, client_node)  # create dataset loader

        return self

    @abstractmethod
    def run_observation(self): 
        pass

    @abstractmethod
    def run_local_training(self):
        pass

    @abstractmethod
    def observation_step(self):
        pass
        
    @abstractmethod
    def local_training_step(self):
        pass

    # ------------------- Reusable GPU-memory helpers -------------------
    def is_offload_weights_to_cpu(self) -> bool:
        """Whether trained weights should be offloaded to CPU after local training.

        Controlled by the strategy arg ``offload_weights_to_cpu`` (default True).
        When True, each client moves its trained weights to CPU and releases GPU
        memory so the next client starts from a clean baseline — essential for
        multi-client runs that would otherwise accumulate one model copy per
        client on the GPU.

        Any client strategy subclass can reuse this flag for consistent
        behaviour across implementations.
        """
        return bool(getattr(self._args, "offload_weights_to_cpu", True))

    # ------------------- Training-complete logging -------------------
    def _log_training_complete(
        self,
        train_record: dict,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a training-complete summary after local training finishes.

        Prints a single-line summary with node id, avg loss, and any extra
        key-value pairs.  Safe no-op when *train_record* is empty.

        Args:
            train_record: Dict returned by ``trainer.train()`` or equivalent.
            extra:        Optional extra fields to include (e.g. r_i, Phi_i).
        """
        node_id = getattr(self._obj, "node_id", "?")
        avg_loss = train_record.get("avg_loss", float("nan"))
        parts = [
            f"Client [{node_id}] training completed",
            f"avg_loss={avg_loss:.6f}" if isinstance(avg_loss, (int, float)) else f"avg_loss={avg_loss}",
        ]
        if extra:
            for k, v in extra.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                elif isinstance(v, list) and len(v) <= 5:
                    parts.append(f"{k}={v}")
                elif isinstance(v, dict):
                    parts.append(f"{k}={{{len(v)} prefixes}}")
                else:
                    parts.append(f"{k}={v}")
        console.info("\n" + " | ".join(parts))

    def offload_weights(self, weights: dict) -> dict:
        """Move a state-dict's tensors to CPU when offloading is enabled.

        Returns the (possibly CPU-resident) weights. When offloading is disabled
        the original dict is returned unchanged so callers can use it
        unconditionally::

            updated_weights = self.offload_weights(updated_weights)

        :param weights: A state-dict mapping names to tensors.
        :return: The weights with tensors on CPU (offload on) or as-is (off).
        """
        if not self.is_offload_weights_to_cpu():
            return weights
        return {
            k: (v.detach().cpu() if hasattr(v, "detach") else v)
            for k, v in weights.items()
        }

    def release_gpu_after_training(self) -> None:
        """Release GPU memory after local training when offloading is enabled.

        Safe no-op when offloading is disabled. Call from a client's
        ``local_training_step`` finally-block after detaching the trained model.
        """
        if self.is_offload_weights_to_cpu():
            GPUMemoryCleaner.release_gpu_memory()

    # ------------------- Convenience: train + auto offload -----------
    def train_and_offload(
        self, trainer: object, epochs: int
    ) -> Tuple[dict, Any]:
        """Train for *epochs* and automatically offload weights to CPU.

        Equivalent to::

            weights, record = trainer.train(epochs)
            weights = self.offload_weights(weights)

        .. note::
            Offloading is controlled by two levels:
            1. YAML config ``offload_weights_to_cpu`` (global default).
            2. Instance attribute ``_auto_offload`` (per‑strategy override,
               default ``True``).  Set to ``False`` to keep weights on GPU
               even when the global flag is on.

        Args:
            trainer: A trainer instance (e.g. ModelTrainer_GLUE).
            epochs:  Number of local epochs.

        Returns:
            ``(weights, record)`` tuple with weights on CPU when offloading
            is active, or on GPU when disabled.
        """
        weights, record = trainer.train(epochs)
        if self._auto_offload:
            weights = self.offload_weights(weights)
        return weights, record

    # ------------------- Centralized GPU cleanup -------------------
    def cleanup_training_resources(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        trainer: Optional[object] = None,
        *,
        verbose: bool = False,
    ) -> None:
        """Release post-training GPU resources in one call.

        Call inside the ``finally`` block of ``local_training_step`` /
        ``observation_step``.  Automatically:

        1. Detaches model / optimizer / EMA from trainer
        2. ``model.cpu()`` → forces GPU tensor deallocation
        3. Clears optimizer gradients & state
        4. ``gc.collect`` + CUDA synchronize + ``empty_cache``

        Args:
            model:     The training model.
            optimizer: The training optimizer.
            trainer:   The trainer instance.
            verbose:   If True, emit debug logs.

        Example::

            try:
                updated_weights, record = tr.train(epochs)
            finally:
                self.cleanup_training_resources(
                    model=training_model,
                    optimizer=optimizer,
                    trainer=tr,
                )
        """
        GPUMemoryCleaner.cleanup_all(
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            verbose=verbose,
        )