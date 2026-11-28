"""
GPU Memory Cleaner — high-level GPU cleanup orchestration.

Composes :class:`ModelUtils` primitives with trainer-aware logic (EMA
shadow teardown, model-parking-to-CPU) to provide a one-stop cleanup
call for FL client post-training.

Usage::

    from flex.ml_utils.gpu_memory_cleaner import GPUMemoryCleaner

    try:
        updated_weights, train_record = tr.train(epochs)
    finally:
        GPUMemoryCleaner.cleanup_all(
            model=training_model,
            optimizer=optimizer,
            trainer=tr,
        )
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch.optim import Optimizer

from .console import console
from .model_utils import ModelUtils


class GPUMemoryCleaner:
    """GPU memory cleanup toolkit — static methods, no instantiation needed."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_all(
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        trainer: Optional[object] = None,
        *,
        verbose: bool = False,
    ) -> None:
        """One-stop cleanup: model + optimizer + trainer state + CUDA cache.

        Call order is carefully designed to avoid dangling references:

        1. Detach model / optimizer from trainer (also clears EMA shadow)
        2. Move model to CPU → force GPU tensor deallocation
        3. Clear optimizer gradients & internal state
        4. Force GC + CUDA cache reclaim (delegates to :meth:`ModelUtils.release_gpu_memory`)

        Args:
            model:     The training model (``.cpu()`` is called before deletion).
            optimizer: The training optimizer.
            trainer:   The trainer instance (model/optimizer/EMA refs are cleared).
            verbose:   If True, emit debug logs (default: silent).
        """
        if trainer is not None:
            GPUMemoryCleaner._detach_trainer(trainer)

        if model is not None:
            GPUMemoryCleaner._cpu_model(model, verbose=verbose)

        if optimizer is not None:
            GPUMemoryCleaner._clear_optimizer(optimizer, verbose=verbose)

        ModelUtils.release_gpu_memory()
        if verbose:
            GPUMemoryCleaner._log_memory("after cleanup")

    @staticmethod
    def release_gpu_memory(*, verbose: bool = False) -> None:
        """Release unused GPU memory (delegates to :meth:`ModelUtils.release_gpu_memory`)."""
        ModelUtils.release_gpu_memory()
        if verbose:
            GPUMemoryCleaner._log_memory("after release")

    @staticmethod
    def detach_trainer(trainer: object) -> None:
        """Detach model / optimizer / EMA from *trainer* so it holds no GPU refs."""
        GPUMemoryCleaner._detach_trainer(trainer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detach_trainer(trainer: object) -> None:
        """Detach model, optimizer, and EMA shadow from a trainer."""
        try:
            trainer.trainer_args.model = None
        except Exception:
            pass
        try:
            trainer.trainer_args.optimizer = None
        except Exception:
            pass

        if hasattr(trainer, "model"):
            try:
                trainer.model = None
            except Exception:
                pass

        if hasattr(trainer, "_ema") and trainer._ema is not None:
            try:
                ema = trainer._ema
                if hasattr(ema, "shadow"):
                    for v in ema.shadow.values():
                        v.data = v.data.cpu()
                    ema.shadow.clear()
            except Exception:
                pass
            try:
                trainer._ema = None
            except Exception:
                pass

    @staticmethod
    def _cpu_model(model: nn.Module, *, verbose: bool = False) -> None:
        """Move *model* to CPU, releasing GPU tensor storage immediately."""
        try:
            model.cpu()
        except Exception as e:
            if verbose:
                console.warn(f"[GPU Memory Cleaner] model.cpu() failed: {e}")

    @staticmethod
    def _clear_optimizer(optimizer: Optimizer, *, verbose: bool = False) -> None:
        """Zero gradients and clear optimizer internal state (momentum buffers)."""
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            if verbose:
                console.warn(f"[GPU Memory Cleaner] zero_grad failed: {e}")
        try:
            optimizer.state.clear()
        except Exception as e:
            if verbose:
                console.warn(f"[GPU Memory Cleaner] state.clear failed: {e}")

    @staticmethod
    def _log_memory(tag: str) -> None:
        """Log current CUDA memory stats at debug level."""
        if not torch.cuda.is_available():
            return
        import torch
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        console.debug(
            f"[GPU Memory Cleaner] {tag}: "
            f"allocated={allocated:.2f}MB | reserved={reserved:.2f}MB"
        )
