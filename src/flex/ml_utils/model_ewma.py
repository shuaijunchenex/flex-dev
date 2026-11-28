from typing import Optional
import torch
import torch.nn as nn


class ModelEWMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated via exponential decay.
    Useful for stable evaluation and as a teacher in self-training setups.

    Parameters
    ----------
    model:
        Source model whose ``state_dict`` is cloned as the initial shadow.
    decay:
        EMA decay rate (default 0.9999).  At each :meth:`update` the shadow
        is blended as ``shadow = decay * shadow + (1 - decay) * model``.
    device:
        Device on which the shadow tensors reside (default CPU).

    Example
    -------
    >>> ema = ModelEWMA(model, decay=0.999, device=torch.device("cuda"))
    >>> for batch in loader:
    ...     # ... training step ...
    ...     ema.update(model)
    >>> ema.apply_to(model)  # restore EMA weights for eval
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: torch.device = torch.device("cpu"),
    ):
        self.decay = decay
        # Detached clone so shadow is independent of the model's computation graph.
        self.shadow = {
            k: v.detach().clone().to(device)
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the current *model* weights into the EMA shadow.

        .. math::
            shadow = decay \\cdot shadow + (1 - decay) \\cdot model
        """
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(d).add_(msd[k], alpha=(1 - d))

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA shadow weights onto *model* (in-place)."""
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)