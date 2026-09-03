"""
SARA-aware model trainer.

Extends the standard trainer to inject slot-level and subspace-level
alignment regularization during local training.

The client strategy sets global anchors, round index, and alignment config
before each local training round via :meth:`set_sara_context`.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..model_trainer import ModelTrainer
from ..model_trainer_args import ModelTrainerArgs
from ...ml_utils import console
from ...ml_utils.model_utils import ModelUtils
from ...ml_algorithms.sara import SARAAlignmentLoss, SARAConfig
from ...ml_algorithms.sara.alignment import collect_lora_pairs


class ModelTrainer_SARA(ModelTrainer):
    """
    Trainer that adds SARA anti-drift alignment regularization.

    Usage (set by client strategy before each round)::

        trainer.set_sara_context(
            global_anchors=global_state_dict,  # server's A_g, B_g
            round_idx=round_idx,
            r_i=local_rank,
            sara_alignment=sara_module,
        )
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = trainer_args.device or ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model
        trainer_args.device = self.device

        # Wrap with DataParallel when multiple CUDA GPUs are available
        self.model = ModelUtils.wrap_data_parallel(self.model, self.device)

        # ── SARA context (set by client strategy) ──
        self._sara_alignment: Optional[SARAAlignmentLoss] = None
        self._global_anchors: Optional[Dict[str, torch.Tensor]] = None
        self._round_idx: int = 0
        self._r_i: int = 0
        self._lora_prefixes: Optional[List[str]] = None  # cached layer prefixes

    # ------------------------------------------------------------------
    # SARA context (called by client strategy before each round)
    # ------------------------------------------------------------------
    def set_sara_context(
        self,
        global_anchors: Dict[str, torch.Tensor],
        round_idx: int,
        r_i: int,
        sara_alignment: SARAAlignmentLoss,
    ) -> None:
        """Set global anchors, round index, client rank, and alignment module."""
        self._global_anchors = {k: v.to(self.device) for k, v in global_anchors.items()}
        self._round_idx = round_idx
        self._r_i = r_i
        self._sara_alignment = sara_alignment
        # Pre-compute LoRA layer prefixes from the global anchors
        self._lora_prefixes = [p for p, _, _ in collect_lora_pairs(self._global_anchors)]

    # ------------------------------------------------------------------
    # Alignment loss computation
    # ------------------------------------------------------------------
    def _compute_alignment_loss(self) -> Tuple[torch.Tensor, float, float]:
        """Compute total alignment loss from current model vs global anchors."""
        if self._sara_alignment is None or self._global_anchors is None:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0

        unwrapped = self._unwrap(self.trainer_args.model)
        param_map: Dict[str, nn.Parameter] = dict(unwrapped.named_parameters())

        total_slot = torch.tensor(0.0, device=self.device)
        total_sub = torch.tensor(0.0, device=self.device)
        count = 0

        for prefix in (self._lora_prefixes or []):
            key_A = f"{prefix}.lora_A"
            key_B = f"{prefix}.lora_B"
            if key_A not in param_map or key_B not in param_map:
                continue
            if key_A not in self._global_anchors or key_B not in self._global_anchors:
                continue

            A_loc = param_map[key_A]
            B_loc = param_map[key_B]
            global_rank = int(self._global_anchors[key_A].shape[0])
            layer_rank = min(
                int(A_loc.shape[0]),
                int(B_loc.shape[1]),
                global_rank,
                int(self._global_anchors[key_B].shape[1]),
            )
            A_glo = self._global_anchors[key_A][: layer_rank, :]  # prefix
            B_glo = self._global_anchors[key_B][:, : layer_rank]  # prefix

            _, slot, sub = self._sara_alignment(
                B_loc[:, :layer_rank],
                A_loc[:layer_rank, :],
                B_glo,
                A_glo,
                layer_rank,
                self._round_idx,
                global_rank=global_rank,
            )
            total_slot = total_slot + slot
            total_sub = total_sub + sub
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0
        return (
            self._sara_alignment.get_lambdas(self._round_idx)[0] * total_slot
            + self._sara_alignment.get_lambdas(self._round_idx)[1] * total_sub,
            total_slot.item() / count,
            total_sub.item() / count,
        )

    # ------------------------------------------------------------------
    # Training step — overrides standard to inject alignment loss
    # ------------------------------------------------------------------
    def train_step(self) -> Dict[str, Any]:
        """Standard training step + SARA alignment regularisation."""
        from ...ml_utils.tqdm_utils import pbar, tqdm_write

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
        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))

        ta.model.to(self.device)
        ta.model.train()

        # Lazy-init metrics if not present (compat with base class)
        if not hasattr(self, "metrics"):
            from ...ml_utils.metric_calculator import MetricCalculator
            self.metrics = MetricCalculator()
        self.metrics.reset()

        # ── Determine alignment lambdas (constant for this step) ──
        lam_slot, lam_sub = 0.0, 0.0
        if self._sara_alignment is not None:
            lam_slot, lam_sub = self._sara_alignment.get_lambdas(self._round_idx)

        loop = pbar(
            train_dl,
            desc=f"SARA train (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=False, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
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
            task_loss = ta.loss_func(outputs, labels)

            # ── SARA alignment regularisation ──
            align_loss = torch.tensor(0.0, device=self.device)
            if self._sara_alignment is not None and self._global_anchors is not None:
                unwrapped = self._unwrap(ta.model)
                # Build direct parameter map (NOT state_dict — state_dict is detached!)
                param_map: Dict[str, nn.Parameter] = dict(unwrapped.named_parameters())
                for prefix in (self._lora_prefixes or []):
                    kA, kB = f"{prefix}.lora_A", f"{prefix}.lora_B"
                    if kA not in param_map or kB not in param_map:
                        continue
                    if kA not in self._global_anchors or kB not in self._global_anchors:
                        continue
                    A_loc = param_map[kA]   # live parameter, gradients flow
                    B_loc = param_map[kB]
                    global_rank = int(self._global_anchors[kA].shape[0])
                    layer_rank = min(
                        int(A_loc.shape[0]),
                        int(B_loc.shape[1]),
                        global_rank,
                        int(self._global_anchors[kB].shape[1]),
                    )
                    A_glo = self._global_anchors[kA][: layer_rank, :]
                    B_glo = self._global_anchors[kB][:, : layer_rank]
                    slot_loss = self._sara_alignment.compute_slot_loss(
                        B_loc[:, :layer_rank],
                        A_loc[:layer_rank, :],
                        B_glo,
                        A_glo,
                        global_rank=global_rank,
                        round_idx=self._round_idx,
                    )
                    sub_loss = self._sara_alignment.compute_subspace_loss(
                        B_loc[:, :layer_rank], B_glo, layer_rank, self._round_idx)
                    align_loss = align_loss + lam_slot * slot_loss + lam_sub * sub_loss

            loss = task_loss + align_loss
            loss.backward()
            ta.optimizer.step()

            loss_scalar = float(loss.item())
            self.metrics.update(loss_scalar, batch_size)

            loop.set_postfix(
                batch=self.metrics.total_batch,
                loss=f"{loss_scalar:.4f}",
                task=f"{float(task_loss.item()):.4f}",
                align=f"{float(align_loss.item()):.6f}",
                avg_loss=f"{self.metrics.avg_loss:.4f}",
            )

        tqdm_write(
            f"[SARA Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={self.metrics.avg_loss:.6f} | batches={self.metrics.total_batch} | "
            f"samples={self.metrics.total_samples} | λ_s={lam_slot:.4f} λ_b={lam_sub:.4f}"
        )

        return self.metrics.get_stats()

    # ------------------------------------------------------------------
    # Other methods — delegate to standard trainer behaviour
    # ------------------------------------------------------------------
    def _eval_initial_loss(self) -> float:
        """Same as standard: loss without gradient."""
        ta = self.trainer_args
        model = ta.model
        model.eval()
        total_loss, total_samples = 0.0, 0
        device = ta.device or self.device
        with torch.no_grad():
            for inputs, labels in ta.train_loader.data_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = model(inputs)
                loss = ta.loss_func(outputs, labels)
                bs = int(inputs.size(0))
                total_loss += float(loss.item()) * bs
                total_samples += bs
        model.train()
        return total_loss / max(total_samples, 1)

    def train(self, epochs: int) -> Any:
        """Standard training loop with alignment."""
        import math
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        ModelUtils.model_training_info(self.trainer_args.model, self.trainer_args.optimizer)
        before_state = {k: v.detach().clone() for k, v in self._unwrap(self.trainer_args.model).state_dict().items()}
        before_weight_l2 = self._state_dict_l2_norm(before_state)
        initial_loss = self._eval_initial_loss()

        train_stats: Dict[str, Any] = {
            "train_loss_sum": 0.0,
            "train_loss_power_two_sum": 0.0,
            "epoch_loss": [],
            "num_batches_sum": 0,
            "num_samples_sum": 0,
            "initial_loss": initial_loss,
        }

        for _ in range(epochs):
            self._epoch_idx += 1
            step_out = self.train_step()
            avg_loss = float(step_out["avg_loss"])
            train_stats["train_loss_sum"] += avg_loss
            train_stats["train_loss_power_two_sum"] += avg_loss ** 2
            train_stats["epoch_loss"].append(avg_loss)
            train_stats["num_batches_sum"] += int(step_out.get("num_batches", 0))
            train_stats["num_samples_sum"] += int(step_out.get("num_samples", 0))

        self._epoch_idx = 0
        train_stats["avg_loss"] = train_stats["train_loss_sum"] / max(epochs, 1)
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])

        after_state = self._unwrap(self.trainer_args.model).state_dict()
        after_weight_l2 = self._state_dict_l2_norm(after_state)
        train_stats["weight_l2_before"] = before_weight_l2
        train_stats["weight_l2_after"] = after_weight_l2
        train_stats["weight_l2_delta"] = self._state_dict_l2_distance(before_state, after_state)

        return self._unwrap(self.trainer_args.model).state_dict(), train_stats

    def observe(self, epochs: int = 5) -> Any:
        """Observation pass (no alignment)."""
        self.trainer_args.total_epochs = epochs
        train_stats = {"train_loss_sum": 0, "epoch_loss": []}
        for _ in range(epochs):
            self._epoch_idx += 1
            step_out = self.train_step()
            train_stats["train_loss_sum"] += float(step_out["avg_loss"])
            train_stats["epoch_loss"].append(float(step_out["avg_loss"]))
        self._epoch_idx = 0
        train_stats["avg_loss"] = train_stats["train_loss_sum"] / max(epochs, 1)
        return self._unwrap(self.trainer_args.model).state_dict(), train_stats

    # ------------------------------------------------------------------
    # L2 helpers (replicated from standard for self-contained file)
    # ------------------------------------------------------------------
    @staticmethod
    def _state_dict_l2_norm(sd: Dict[str, torch.Tensor]) -> float:
        return math.sqrt(sum(v.detach().float().norm().item() ** 2 for v in sd.values()))

    @staticmethod
    def _state_dict_l2_distance(sd1: Dict, sd2: Dict) -> float:
        d = 0.0
        for k in sd1:
            if k in sd2:
                d += (sd1[k].detach().float() - sd2[k].detach().float()).norm().item() ** 2
        return math.sqrt(d)

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, nn.DataParallel) else model
