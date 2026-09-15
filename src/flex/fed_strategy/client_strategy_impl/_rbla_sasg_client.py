"""
RBLA-SASG client training strategy.

Receives semantic slots from server, initializes local LoRA from those slots,
trains locally (with optional semantic anchoring), and uploads results
including r_i and Phi_i.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from flex.fed_strategy.strategy_args import StrategyArgs
from ..client_strategy import ClientStrategy
from ...ml_utils.model_utils import ModelUtils
from ...ml_utils import console
from ...ml_utils.metric_calculator import MetricCalculator
from ...ml_utils.training_utils import TrainingUtils
from ...fed_node.fed_node_vars import FedNodeVars
from ...ml_algorithms.rblasa.semantic_grid import (
    semantic_grid_mapping,
    make_slot_importance_weights,
    alignment_lambda,
    semantic_anchoring_loss,
)


class RblaSasgClientTrainingStrategy(ClientStrategy):
    """
    RBLA-SASG client strategy.

    Extends the standard RBLA client pattern with:
    - Receiving per-client semantic slots from server
    - Initializing local LoRA from those slots
    - Computing semantic anchoring loss during training
    - Uploading r_i and Phi_i alongside weights

    Config keys (via config_dict["rbla_sasg"] or top-level):
        lambda_0:    Initial alignment strength (default 0.0 = no anchoring).
        lambda_min:  Minimum alignment strength (default 0.0).
        beta:        Alignment decay rate (default 0.0).
        alpha:       Slot importance skew (default 0.0 = uniform).
    """

    def __init__(self, args, client_node):
        super().__init__()
        self._args = args
        self._strategy_type = "rbla_sasg"
        self._obj = client_node

        # SASG-specific state
        self._r_i: int = 0
        self._Phi_i: List[int] = []
        self._global_A_snapshot: dict = {}  # prefix -> A_g tensor
        self._global_B_snapshot: dict = {}  # prefix -> B_g tensor

        # Per-prefix metadata (populated by receive_sasg_slots)
        self._rank_by_prefix: Dict[str, int] = {}
        self._Phi_by_prefix: Dict[str, List[int]] = {}

        # Alignment config
        cfg = client_node.node_var.config_dict if client_node.node_var else {}
        sasg_cfg = cfg.get("rbla_sasg", {})
        self._lambda_0 = float(sasg_cfg.get("lambda_0", 0.0))
        self._lambda_min = float(sasg_cfg.get("lambda_min", 0.0))
        self._beta = float(sasg_cfg.get("beta", 0.0))
        self._alpha = float(sasg_cfg.get("alpha", 0.0))
        self._round_counter: int = 0

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "rbla_sasg"
        self._obj = client_node
        return

    # ------------------------------------------------------------------
    # SASG-specific receive
    # ------------------------------------------------------------------
    def receive_sasg_slots(
        self,
        sliced_weight: dict,
        r_i: int,
        Phi_i: List[int],
        rank_by_prefix: Dict[str, int] | None = None,
        Phi_by_prefix: Dict[str, List[int]] | None = None,
    ) -> None:
        """
        Receive per-client semantic slots from server.

        Args:
            sliced_weight:  state_dict with lora_A/lora_B sliced per prefix.
            r_i:            Global max local LoRA rank (backward compat).
            Phi_i:          Global semantic grid mapping (backward compat).
            rank_by_prefix: Per-prefix local rank {prefix: r}.
            Phi_by_prefix:  Per-prefix semantic grid {prefix: [slot, ...]}.
        """
        self._r_i = r_i
        self._Phi_i = list(Phi_i)

        # Store per-prefix metadata
        if rank_by_prefix:
            self._rank_by_prefix = dict(rank_by_prefix)
        if Phi_by_prefix:
            self._Phi_by_prefix = {p: list(v) for p, v in Phi_by_prefix.items()}

        # Save global slot snapshots for anchoring loss
        node_var = self._obj.node_var
        if node_var and node_var.model:
            from flex.ml_algorithms.lora.lora_utils import LoRAUtils
            rank_dict = LoRAUtils.get_lora_ranks(node_var.model)
            # The sliced_weight has A: [r_i, in], B: [out, r_i] — these ARE the slots
            for key, tensor in sliced_weight.items():
                if key.endswith(".lora_A"):
                    prefix = key.rsplit(".lora_A", 1)[0]
                    self._global_A_snapshot[prefix] = tensor.detach().clone()
                elif key.endswith(".lora_B"):
                    prefix = key.rsplit(".lora_B", 1)[0]
                    self._global_B_snapshot[prefix] = tensor.detach().clone()

        # Load sliced weight as model_weight
        node_var.model_weight = sliced_weight

    def receive_weight(self, global_weight: dict) -> None:
        """Fallback: standard receive (used when server doesn't call receive_sasg_slots)."""
        self._obj.node_var.cache_weight = global_weight

    def set_local_weight(self) -> None:
        """Standard set_local_weight — delegates to cached weight."""
        if hasattr(self._obj.node_var, "cache_weight") and self._obj.node_var.cache_weight:
            self._obj.node_var.model_weight = self._obj.node_var.cache_weight

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def run_observation(self) -> dict:
        print(f"\n Observation Client [{self._obj.node_id}] ...\n")
        _, train_record = self.observation_step()
        return {
            "node_id": self._obj.node_id,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num,
        }

    def observation_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = node_vars.device if hasattr(node_vars, "device") else "cpu"

        observe_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        observe_model.load_state_dict(node_vars.model_weight, strict=True)
        optimizer = node_vars.optimizer_builder.rebuild(observe_model.parameters())
        ModelUtils.clear_all(observe_model, optimizer)

        tr = node_vars.trainer
        orig_model = tr.trainer_args.model
        orig_optimizer = tr.trainer_args.optimizer
        orig_device = getattr(tr.trainer_args, "device", None)
        try:
            tr.set_model(observe_model)
            tr.set_optimizer(optimizer)
            tr.trainer_args.device = device
            local_epochs = int(cfg.get("training", {}).get("local_epochs", 1))
            updated_weights, train_record = self._train_with_anchoring(
                tr, local_epochs, device
            )
        finally:
            if orig_model is not None:
                tr.set_model(orig_model)
            if orig_optimizer is not None:
                tr.set_optimizer(orig_optimizer)
            if orig_device is not None:
                tr.trainer_args.device = orig_device
            ModelUtils.release_gpu_memory()

        return updated_weights, train_record

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------
    def run_local_training(self) -> dict:
        updated_weights, train_record = self.local_training_step()
        self._round_counter += 1
        self._log_training_complete(
            train_record,
            extra={"r_i": self._r_i, "Phi_len": len(self._Phi_i)},
        )
        return updated_weights, {
            "node_id": self._obj.node_id,
            "updated_weights": updated_weights,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num,
            "r_i": self._r_i,
            "Phi_i": list(self._Phi_i),
            "rank_by_prefix": dict(self._rank_by_prefix),
            "Phi_by_prefix": {p: list(v) for p, v in self._Phi_by_prefix.items()},
        }

    def local_training_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        training_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)

        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())
        ModelUtils.clear_all(training_model, optimizer)

        tr = node_vars.trainer
        tr.set_model(training_model)
        tr.set_optimizer(optimizer)
        tr.trainer_args.device = device

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        try:
            updated_weights, train_record = self._train_with_anchoring(
                tr, local_epochs, device
            )
        finally:
            self.cleanup_training_resources(
                model=training_model,
                optimizer=optimizer,
                trainer=tr,
            )

        node_vars.model_weight = updated_weights
        return copy.deepcopy(updated_weights), train_record

    # ------------------------------------------------------------------
    # Training with optional semantic anchoring
    # ------------------------------------------------------------------
    def _train_with_anchoring(
        self, trainer, epochs: int, device: str
    ) -> Tuple[dict, Any]:
        """
        Run local training with optional semantic anchoring loss.

        Anchoring is skipped when lambda == 0 or no global slot snapshot.
        """
        lambda_t = alignment_lambda(
            self._round_counter, self._lambda_0, self._lambda_min, self._beta
        )

        if lambda_t <= 0.0 or not self._Phi_i or not self._global_A_snapshot:
            # No anchoring — delegate to standard trainer
            weights, record = trainer.train(epochs)
            return self.offload_weights(weights), record

        # ── Anchoring path ──────────────────────────────────────────
        from flex.ml_utils.tqdm_utils import pbar, tqdm_write

        ta = trainer.trainer_args
        ta.model.to(device)
        ta.model.train()
        ta.total_epochs = epochs
        trainer._epoch_idx = 0

        ModelUtils.model_training_info(ta.model, ta.optimizer)
        before_state = {
            k: v.detach().clone()
            for k, v in trainer._unwrap(ta.model).state_dict().items()
        }
        before_weight_l2 = trainer._state_dict_l2_norm(before_state)

        # --- Pre-compute once (outside batch loop) ---
        # 1. Slot importance weights.  Use the largest received global slot so
        # adaptive Phi_by_prefix gets the same weighting semantics as aggregation.
        max_rank = max(
            [max(self._Phi_i)]
            + [max(phi) for phi in self._Phi_by_prefix.values() if phi]
        )
        omega = make_slot_importance_weights(max_rank, self._alpha)

        # 2. Build direct parameter lookup: prefix -> (lora_A_param, lora_B_param)
        #    Avoids calling state_dict() every batch.
        lora_param_map: dict = {}  # prefix -> [A_param, B_param]
        for name, param in ta.model.named_parameters():
            if name.endswith(".lora_A"):
                prefix = name[: -len(".lora_A")]
                lora_param_map.setdefault(prefix, [None, None])[0] = param
            elif name.endswith(".lora_B"):
                prefix = name[: -len(".lora_B")]
                lora_param_map.setdefault(prefix, [None, None])[1] = param

        # 3. Move global snapshots to device once + cache per-prefix Phi.
        #
        # The server broadcasts sliced A/B tensors ordered by Phi_by_prefix.  For
        # anchoring, rebuild a sparse global-slot view when that metadata is
        # available, so the same global slot ids are used by broadcast, anchor,
        # and aggregation.  If metadata is missing or invalid, fall back to the
        # previous local contiguous view.
        global_A_dev: dict = {}
        global_B_dev: dict = {}
        phi_cache: dict = {}       # prefix -> phi list
        for prefix in self._global_A_snapshot:
            if prefix not in lora_param_map:
                continue
            A_g = self._global_A_snapshot[prefix].to(device)
            B_g = self._global_B_snapshot[prefix].to(device)
            r_i_local = lora_param_map[prefix][0].shape[0]
            phi_prefix = self._Phi_by_prefix.get(prefix)

            if self._is_valid_anchor_phi(phi_prefix, A_g, B_g, r_i_local):
                phi = [int(s) for s in phi_prefix]
                R_g = max(max(phi), A_g.shape[0], B_g.shape[1])
                A_full = A_g.new_zeros((R_g, A_g.shape[1]))
                B_full = B_g.new_zeros((B_g.shape[0], R_g))

                for k_local, s_global in enumerate(phi):
                    idx = s_global - 1
                    A_full[idx, :] = A_g[k_local, :]
                    B_full[:, idx] = B_g[:, k_local]

                global_A_dev[prefix] = A_full
                global_B_dev[prefix] = B_full
                phi_cache[prefix] = phi
            else:
                if phi_prefix:
                    console.warn(
                        f"[RBLA-SASG] Invalid Phi_by_prefix for '{prefix}' "
                        f"(phi_len={len(phi_prefix)}, A={tuple(A_g.shape)}, "
                        f"B={tuple(B_g.shape)}, local_rank={r_i_local}); "
                        "falling back to local contiguous anchoring."
                    )
                global_A_dev[prefix] = A_g
                global_B_dev[prefix] = B_g
                R_g = A_g.shape[0]
                phi_cache[prefix] = semantic_grid_mapping(min(r_i_local, R_g), R_g)

        # 4. Build list of anchor entries (prefix, lora_A_param, lora_B_param, A_g, B_g, phi)
        anchor_entries = []
        for prefix, (a_param, b_param) in lora_param_map.items():
            if prefix not in global_A_dev:
                continue
            anchor_entries.append((
                prefix,
                a_param, b_param,
                global_A_dev[prefix], global_B_dev[prefix],
                phi_cache[prefix],
            ))

        train_dl = ta.train_loader.data_loader
        total_epochs = getattr(ta, "total_epochs", epochs)

        stats: dict = {
            "train_loss_sum": 0.0,
            "epoch_loss": [],
            "train_loss_power_two_sum": 0.0,
            "keras_train_loss_sum": 0.0,
            "keras_epoch_loss": [],
            "keras_train_loss_power_two_sum": 0.0,
            "task_epoch_loss": [],
            "anchor_epoch_loss": [],
            "num_batches_sum": 0,
            "num_samples_sum": 0,
            "initial_loss": 0.0,
            "lambda_t": lambda_t,
        }

        if not anchor_entries:
            weights, record = trainer.train(epochs)
            return self.offload_weights(weights), record

        for _ in range(epochs):
            trainer._epoch_idx += 1
            metrics = MetricCalculator()
            task_metrics = MetricCalculator()
            anchor_metrics = MetricCalculator()

            loop = pbar(
                train_dl,
                desc=f"Training (epoch {trainer._epoch_idx}"
                     f"{'/' + str(total_epochs) if total_epochs else ''})",
                leave=False, ncols=120, mininterval=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            )
            for inputs, labels in loop:
                inputs = TrainingUtils.to_device(inputs, device)
                labels = TrainingUtils.to_device(labels, device).long()
                self._validate_labels(labels, ta)

                try:
                    batch_size = int(inputs.size(0))
                except Exception:
                    batch_size = int(labels.size(0))

                ta.optimizer.zero_grad(set_to_none=True)

                amp_enabled = bool(getattr(trainer, "amp_enabled", False))
                amp_dtype = getattr(trainer, "_amp_dtype", None)
                with TrainingUtils.make_autocast(device, amp_enabled, amp_dtype):
                    outputs = ta.model(inputs)
                    task_loss = ta.loss_func(outputs, labels)

                    # Compute semantic anchoring loss (vectorized, O(1) tensor ops)
                    anchor_loss = torch.tensor(0.0, device=device)
                    for _, A_param, B_param, A_g, B_g, phi in anchor_entries:
                        anchor_loss = anchor_loss + semantic_anchoring_loss(
                            A_param, B_param, A_g, B_g, phi, omega,
                        )

                    loss = task_loss + lambda_t * anchor_loss
                loss.backward()
                ta.optimizer.step()

                loss_scalar = float(loss.item())
                task_scalar = float(task_loss.item())
                anchor_scalar = float(anchor_loss.item())
                metrics.update(loss_scalar, batch_size)
                task_metrics.update(task_scalar, batch_size)
                anchor_metrics.update(anchor_scalar, batch_size)

                loop.set_postfix(
                    batch=metrics.total_batch,
                    loss=f"{loss_scalar:.4f}",
                    task=f"{task_scalar:.4f}",
                    anchor=f"{anchor_scalar:.4f}",
                    avg_loss=f"{metrics.avg_loss:.4f}",
                    avg_loss_keras=f"{metrics.keras_loss:.4f}",
                    lr=ta.optimizer.param_groups[0]["lr"],
                )

            tqdm_write(
                f"[SASG Epoch {trainer._epoch_idx}"
                f"{'/' + str(total_epochs) if total_epochs else ''} Finished] "
                f"avg_loss={metrics.avg_loss:.6f} | "
                f"keras_loss={metrics.keras_loss:.6f} | "
                f"task={task_metrics.avg_loss:.6f} | "
                f"anchor={anchor_metrics.avg_loss:.6f} | "
                f"batches={metrics.total_batch} | "
                f"samples={metrics.total_samples} | "
                f"lambda={lambda_t:.6f} | device={ta.device}"
            )

            avg_loss = float(metrics.avg_loss)
            keras_loss = float(metrics.keras_loss)
            stats["train_loss_sum"] += avg_loss
            stats["train_loss_power_two_sum"] += avg_loss ** 2
            stats["epoch_loss"].append(avg_loss)
            stats["keras_train_loss_sum"] += keras_loss
            stats["keras_train_loss_power_two_sum"] += keras_loss ** 2
            stats["keras_epoch_loss"].append(keras_loss)
            stats["task_epoch_loss"].append(float(task_metrics.avg_loss))
            stats["anchor_epoch_loss"].append(float(anchor_metrics.avg_loss))
            stats["num_batches_sum"] += int(metrics.total_batch)
            stats["num_samples_sum"] += int(metrics.total_samples)

        trainer._epoch_idx = 0
        stats["avg_loss"] = stats["train_loss_sum"] / max(epochs, 1)
        stats["keras_avg_loss"] = stats["keras_train_loss_sum"] / max(epochs, 1)
        stats["sqrt_train_loss_power_two_sum"] = math.sqrt(
            stats["train_loss_power_two_sum"]
        )
        stats["keras_sqrt_train_loss_power_two_sum"] = math.sqrt(
            stats["keras_train_loss_power_two_sum"]
        )

        after_state = trainer._unwrap(ta.model).state_dict()
        after_weight_l2 = trainer._state_dict_l2_norm(after_state)
        stats["weight_l2_before"] = before_weight_l2
        stats["weight_l2_after"] = after_weight_l2
        stats["weight_l2_delta"] = trainer._state_dict_l2_distance(
            before_state, after_state
        )
        if hasattr(trainer, "_state_dict_l2_distance_layerwise"):
            stats["weight_l2_delta_keras"] = trainer._state_dict_l2_distance_layerwise(
                before_state, after_state
            )

        updated_weights = copy.deepcopy(after_state)
        updated_weights = self.offload_weights(updated_weights)
        return updated_weights, stats

    @staticmethod
    def _validate_labels(labels: torch.Tensor, trainer_args) -> None:
        min_label = labels.min().item()
        max_label = labels.max().item()
        num_classes = None
        if (
            trainer_args.loss_func is not None
            and hasattr(trainer_args.loss_func, "weight")
            and trainer_args.loss_func.weight is not None
        ):
            num_classes = trainer_args.loss_func.weight.size(0)
        elif (
            hasattr(trainer_args.model, "config")
            and hasattr(trainer_args.model.config, "num_labels")
        ):
            num_classes = trainer_args.model.config.num_labels

        if num_classes is not None and (min_label < 0 or max_label >= num_classes):
            unique_labels = torch.unique(labels).cpu().numpy().tolist()
            console.error(
                f"[Label Range Error] min={min_label}, max={max_label}, "
                f"num_classes={num_classes}, unique_labels={unique_labels}. "
                "In extreme non-IID, each client should have all label types."
            )
            raise ValueError(
                f"Invalid label values: min={min_label}, max={max_label}, "
                f"but num_classes={num_classes}"
            )

    @staticmethod
    def _is_valid_anchor_phi(
        phi: List[int] | None,
        A_slice: torch.Tensor,
        B_slice: torch.Tensor,
        local_rank: int,
    ) -> bool:
        if not phi:
            return False
        if len(phi) != local_rank:
            return False
        if len(phi) != A_slice.shape[0] or len(phi) != B_slice.shape[1]:
            return False
        try:
            slots = [int(s) for s in phi]
        except (TypeError, ValueError):
            return False
        if any(s < 1 for s in slots):
            return False
        return len(set(slots)) == len(slots)
