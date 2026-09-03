import torch
import gc
from typing import Optional
from .console import console
from torch.optim import Optimizer
from torch import nn

class ModelUtils:
    @staticmethod
    def accelerator_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        console.debug(f"Using device: {device}")
        return device

    @staticmethod
    def wrap_data_parallel(model: nn.Module, device: torch.device | str) -> nn.Module:
        """
        Wrap model with nn.DataParallel when multiple CUDA GPUs are available.

        Rules:
        - Only activates on CUDA with >= 2 GPUs.
        - MPS / CPU are single-device only; wrapping is skipped.
        - If already wrapped, returns the model as-is.

        Returns:
            The (possibly wrapped) model, already moved to *device*.
        """
        device = torch.device(device)
        model = model.to(device)

        if (
            device.type == "cuda"
            and torch.cuda.device_count() > 1
            and not isinstance(model, nn.DataParallel)
        ):
            gpu_count = torch.cuda.device_count()
            model = nn.DataParallel(model)
            console.ok(f"[DataParallel] Enabled: {gpu_count} GPUs detected.")
        return model

    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        """
        Return the underlying model, stripping nn.DataParallel if present.
        Use this before calling state_dict() for FL aggregation.
        """
        if isinstance(model, nn.DataParallel):
            return model.module
        return model

    @staticmethod
    def clear_all(model: nn.Module, optimizer: Optimizer, reset_optimizer: bool = True):
        """
        Clears gradients and releases unused cached GPU memory.
        Optionally resets the optimizer state (momentum buffers etc.).

        Args:
            reset_optimizer: If True (default), also clears the optimizer state.
                             Set to False when the optimizer state should be
                             preserved across rounds (e.g. persistent momentum).
        """
        ModelUtils.clear_model_grads(model)
        if reset_optimizer:
            ModelUtils.reset_optimizer_state(optimizer)
        ModelUtils.clear_cuda_cache()

    @staticmethod
    def model_training_info(model: nn.Module, optimizer: Optimizer):
        """
        Logs model and optimizer training information.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = trainable_params / total_params if total_params > 0 else 0.0

        def format_p(n):
            if n >= 1e9: return f"{n/1e9:.3f}B"
            if n >= 1e6: return f"{n/1e6:.3f}M"
            return f"{n:,}"

        total_fmt = format_p(total_params)
        trainable_fmt = format_p(trainable_params)

        console.ok(
            f"[Training Info] {model.__class__.__name__} | "
            f"trainable={trainable_fmt} ({trainable_params:,}) / "
            f"total={total_fmt} ({total_params:,}) | "
            f"ratio={ratio:.2%} | "
            f"optimizer={optimizer.__class__.__name__}"
        )

    @staticmethod
    def restore_optimizer_state(optimizer: Optimizer, saved_state: dict | None, device: str | torch.device) -> None:
        """Load a previously saved optimizer state_dict into *optimizer* and remap
        all state tensors to *device*.

        If *saved_state* is ``None`` (first round, cold start) the optimizer is
        left with an empty state.  If the saved state is stale or mismatched (e.g.
        after aggregation changed the model shape) the state is silently discarded
        so training can continue from a cold start.

        Args:
            optimizer:   The freshly-built optimizer whose state should be restored.
            saved_state: The ``state_dict`` returned by a previous call to
                         :meth:`snapshot_optimizer_state`, or ``None``.
            device:      Training device; all state tensors will be moved here.
        """
        if saved_state is None:
            return
        try:
            optimizer.load_state_dict(saved_state)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception:
            # Shape mismatch after aggregation → discard stale state.
            optimizer.state.clear()

    @staticmethod
    def snapshot_optimizer_state(optimizer: Optimizer) -> dict:
        """Capture the current optimizer state as a CPU-resident snapshot.

        The returned dict can be stored in ``node_var.persistent_optimizer_state``
        and later passed to :meth:`restore_optimizer_state` to resume momentum /
        adaptive-LR accumulation in the next round.

        All tensors are moved to CPU so the snapshot survives model disposal and
        device transitions between rounds.

        Args:
            optimizer: The optimizer whose state should be snapshotted.

        Returns:
            A ``state_dict`` with all tensors on CPU.
        """
        state_to_save = optimizer.state_dict()
        for state in state_to_save.get("state", {}).values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        return state_to_save

    @staticmethod
    def clear_model_grads(model: nn.Module):
        """
        Clears the gradients of all parameters in the given model by setting .grad to None.
        """
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

    @staticmethod
    def clear_cuda_cache():
        """
        Releases unused cached GPU memory to help avoid memory accumulation.
        """
        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        console.info(f"[Cuda Cache Cleared] allocated={allocated:.2f}MB | reserved={reserved:.2f}MB")

    @staticmethod
    def release_gpu_memory():
        """
        Release all unused GPU memory and run garbage collection.
        Silent version — no console output.  Suitable for calling between
        client training rounds to prevent memory accumulation.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            console.debug(
                f"[GPU Memory Released] allocated={allocated:.2f}MB | "
                f"reserved={reserved:.2f}MB"
            )

    @staticmethod
    def cleanup_training_resources(
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        trainer: Optional[object] = None,
        *,
        verbose: bool = False,
    ) -> None:
        """Release post-training GPU resources in one call.

        Delegates to ``GPUMemoryCleaner.cleanup_all``.  Automatically:
        detach trainer → clear EMA → model.cpu() → clear optimizer → CUDA reclaim.

        Args:
            model:     The training model.
            optimizer: The training optimizer.
            trainer:   The trainer instance.
            verbose:   If True, emit debug logs.
        """
        from .gpu_memory_cleaner import GPUMemoryCleaner
        GPUMemoryCleaner.cleanup_all(
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            verbose=verbose,
        )

    @staticmethod
    def reset_optimizer_state(optimizer: Optimizer):
        """
        Clears the internal state of an optimizer (e.g., momentum buffers),
        and outputs the current learning rates.
        """
        optimizer.state.clear()
        num_params = sum(
            p.numel() for group in optimizer.param_groups for p in group['params']
        )

        lrs = [group.get("lr", None) for group in optimizer.param_groups]

        console.info(
            f"[Optimizer Reset] {optimizer.__class__.__name__} "
            f"| id={id(optimizer)} | params={num_params:,} | lr={lrs}"
        )

