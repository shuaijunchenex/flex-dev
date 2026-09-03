from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

import torch
import torch.nn as nn
import numpy as np
from ...ml_utils.tqdm_utils import pbar, tqdm_write

from flex.model_trainer.model_trainer_args import ModelTrainerArgs

from ..model_trainer import ModelTrainer
from ...ml_utils import console
from ...ml_utils.model_utils import ModelUtils
from ...ml_utils.metric_calculator import MetricCalculator


class ModelTrainer_LoraGrad(ModelTrainer):
    """
    Standard trainer that additionally accumulates the **real** gradients of
    every ``lora_A`` / ``lora_B`` parameter across all batches and epochs.

    After ``train()`` the returned ``train_stats`` dict contains::

        "lora_grad": {
            "<layer_prefix>": {
                "lora_A": np.ndarray,   # mean gradient for lora_A (same shape)
                "lora_B": np.ndarray,   # mean gradient for lora_B (same shape)
            },
            ...
        }

    Implementation notes
    --------------------
    * ``retain_grad()`` is called once per parameter so that gradients on
      non-leaf tensors (LoRA matrices inside DataParallel, etc.) are kept.
    * Gradients are accumulated **as sums** after every ``loss.backward()``
      call, before ``optimizer.step()`` clears them (they are not zeroed by
      step; only ``zero_grad()`` zeros them, and we read before that).
    * The final value is divided by the total number of backward passes
      (batches × epochs) to give a per-sample-normalised mean.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, trainer_args: ModelTrainerArgs) -> None:
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = trainer_args.device or ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model
        trainer_args.device = self.device

        self.model = ModelUtils.wrap_data_parallel(self.model, self.device)
        trainer_args.model = self.model
        self.metrics = MetricCalculator()

        # Accumulated gradient sums: { prefix -> { "lora_A": Tensor, "lora_B": Tensor } }
        self._grad_accum: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        self._backward_count: int = 0

    # ------------------------------------------------------------------
    # Model setter
    # ------------------------------------------------------------------
    def set_model(self, model: nn.Module) -> "ModelTrainer_LoraGrad":
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = ModelUtils.wrap_data_parallel(self.trainer_args.model, self.device)
        self.trainer_args.model = self.model
        return self

    # ------------------------------------------------------------------
    # Internal: collect lora_A / lora_B named parameters
    # ------------------------------------------------------------------
    def _collect_lora_params(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Walk the (possibly DataParallel-wrapped) model and return::

            { layer_prefix -> { "lora_A": param, "lora_B": param } }
        """
        base = self._unwrap(self.trainer_args.model)
        result: Dict[str, Dict[str, nn.Parameter]] = {}
        a_map: Dict[str, nn.Parameter] = {}
        b_map: Dict[str, nn.Parameter] = {}

        for name, param in base.named_parameters():
            if name.endswith(".lora_A"):
                prefix = name[: -len(".lora_A")]
                a_map[prefix] = param
            elif name.endswith(".lora_B"):
                prefix = name[: -len(".lora_B")]
                b_map[prefix] = param

        for prefix in set(a_map) | set(b_map):
            result[prefix] = {}
            if prefix in a_map:
                result[prefix]["lora_A"] = a_map[prefix]
            if prefix in b_map:
                result[prefix]["lora_B"] = b_map[prefix]

        return result

    # ------------------------------------------------------------------
    # Gradient accumulation helpers
    # ------------------------------------------------------------------
    def _reset_grad_accum(self) -> None:
        self._grad_accum = {}
        self._backward_count = 0

    def _accumulate_grads(
        self, lora_params: Dict[str, Dict[str, nn.Parameter]]
    ) -> None:
        """Read .grad of each lora param and add to running sum."""
        for prefix, param_dict in lora_params.items():
            if prefix not in self._grad_accum:
                self._grad_accum[prefix] = {"lora_A": None, "lora_B": None}
            for key, param in param_dict.items():
                if param.grad is None:
                    continue
                g = param.grad.detach().float().cpu()
                prev = self._grad_accum[prefix][key]
                self._grad_accum[prefix][key] = g if prev is None else prev + g

        self._backward_count += 1

    def _finalize_grads(self) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """Divide accumulated sums by backward count → mean gradient."""
        if self._backward_count == 0:
            return {}
        result: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for prefix, gd in self._grad_accum.items():
            result[prefix] = {
                key: (gd[key] / self._backward_count).numpy()
                if gd[key] is not None
                else None
                for key in ("lora_A", "lora_B")
            }
        return result

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------
    def train_step(self) -> Dict[str, Any]:
        ta = self.trainer_args
        if ta.optimizer is None:
            raise ValueError("Trainer optimizer is None.")
        if ta.model is None:
            raise ValueError("Trainer model is None.")
        if ta.loss_func is None:
            raise ValueError("Trainer loss function is None.")
        if ta.train_loader is None:
            raise ValueError("Trainer train_loader is None.")

        train_dl = ta.train_loader.data_loader
        if not hasattr(train_dl, "__iter__"):
            raise TypeError(
                f"train_loader must be an iterable DataLoader, got {type(train_dl).__name__}"
            )

        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))
        ta.model.to(self.device)
        ta.model.train()
        self.metrics.reset()

        # Collect LoRA params and enable retain_grad for non-leaf tensors
        lora_params = self._collect_lora_params()
        for param_dict in lora_params.values():
            for param in param_dict.values():
                if not param.is_leaf:
                    param.retain_grad()

        loop = pbar(
            train_dl,
            desc=(
                f"Training (epoch {self._epoch_idx}"
                f"{'/' + str(total_epochs) if total_epochs else ''})"
            ),
            leave=False,
            ncols=120,
            mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for inputs, labels in loop:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()

            try:
                batch_size = int(inputs.size(0))
            except Exception:
                batch_size = int(labels.size(0))

            ta.optimizer.zero_grad()
            outputs = ta.model(inputs)
            loss = ta.loss_func(outputs, labels)
            loss.backward()

            # --- Accumulate real gradients BEFORE optimizer.step() ---
            self._accumulate_grads(lora_params)

            ta.optimizer.step()

            loss_scalar = float(loss.item())
            self.metrics.update(loss_scalar, batch_size)
            loop.set_postfix(
                batch=self.metrics.total_batch,
                loss=f"{loss_scalar:.4f}",
                avg_loss=f"{self.metrics.avg_loss:.4f}",
                avg_loss_keras=f"{self.metrics.keras_loss:.4f}",
                lr=ta.optimizer.param_groups[0]["lr"],
            )

        tqdm_write(
            f"[Epoch {self._epoch_idx}"
            f"{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={self.metrics.avg_loss:.6f} | keras_loss={self.metrics.keras_loss:.6f} | "
            f"batches={self.metrics.total_batch} | samples={self.metrics.total_samples} | "
            f"device={ta.device}"
        )

        return self.metrics.get_stats()

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    def train(self, epochs: int) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        ModelUtils.model_training_info(self.model, self.trainer_args.optimizer)
        before_state = {
            k: v.detach().clone()
            for k, v in self._unwrap(self.trainer_args.model).state_dict().items()
        }
        before_weight_l2 = self._state_dict_l2_norm(before_state)

        self._reset_grad_accum()

        train_stats: Dict[str, Any] = {
            "train_loss_sum": 0.0,
            "train_loss_power_two_sum": 0.0,
            "epoch_loss": [],
            "keras_train_loss_sum": 0.0,
            "keras_train_loss_power_two_sum": 0.0,
            "keras_epoch_loss": [],
            "num_batches_sum": 0,
            "num_samples_sum": 0,
        }

        for _ in range(epochs):
            self._epoch_idx += 1
            step_out = self.train_step()

            avg_loss = float(step_out["avg_loss"])
            keras_loss = float(step_out["keras_loss"])
            num_batches = int(step_out.get("num_batches", 0))
            num_samples = int(step_out.get("num_samples", 0))

            train_stats["train_loss_sum"] += avg_loss
            train_stats["train_loss_power_two_sum"] += avg_loss ** 2
            train_stats["epoch_loss"].append(avg_loss)
            train_stats["keras_train_loss_sum"] += keras_loss
            train_stats["keras_train_loss_power_two_sum"] += keras_loss ** 2
            train_stats["keras_epoch_loss"].append(keras_loss)
            train_stats["num_batches_sum"] += num_batches
            train_stats["num_samples_sum"] += num_samples

        self._epoch_idx = 0

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / max(epochs, 1)
        train_stats["keras_avg_loss"] = (
            train_stats["keras_train_loss_sum"] / max(epochs, 1)
        )
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(
            train_stats["train_loss_power_two_sum"]
        )
        train_stats["keras_sqrt_train_loss_power_two_sum"] = math.sqrt(
            train_stats["keras_train_loss_power_two_sum"]
        )

        after_state = self._unwrap(self.trainer_args.model).state_dict()
        after_weight_l2 = self._state_dict_l2_norm(after_state)
        train_stats["weight_l2_before"] = before_weight_l2
        train_stats["weight_l2_after"] = after_weight_l2
        train_stats["weight_l2_delta"] = self._state_dict_l2_distance(
            before_state, after_state
        )
        train_stats["weight_l2_delta_keras"] = self._state_dict_l2_distance_layerwise(
            before_state, after_state
        )

        # --- Attach mean gradients ---
        train_stats["lora_grad"] = self._finalize_grads()

        return self._unwrap(self.trainer_args.model).state_dict(), train_stats
