from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ._model_trainer_sara import ModelTrainer_SARA
from ..model_trainer_args import ModelTrainerArgs


class ModelTrainer_AdaptiveSARA(ModelTrainer_SARA):
    """SARA trainer with rank-adaptive alignment strength.

    For each LoRA layer, the SARA regularization is scaled by a client/layer
    rank weight:

        w = min + (max - min) * (local_rank / global_rank) ** gamma

    Higher-rank local LoRA matrices therefore receive stronger alignment, while
    lower-rank matrices receive weaker alignment.
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

    def _rank_adaptive_weight(self, local_rank: int, global_rank: int) -> float:
        cfg = getattr(self._sara_alignment, "cfg", None)
        if cfg is None:
            return 1.0

        min_w = float(getattr(cfg, "rank_weight_min", 0.5))
        max_w = float(getattr(cfg, "rank_weight_max", 1.5))
        gamma = float(getattr(cfg, "rank_weight_gamma", 1.0))
        if global_rank <= 0:
            return 1.0

        ratio = max(0.0, min(1.0, float(local_rank) / float(global_rank)))
        return min_w + (max_w - min_w) * (ratio ** gamma)

    def train_step(self) -> Dict[str, Any]:
        """Standard SARA training step with rank-adaptive alignment weight."""
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

        if not hasattr(self, "metrics"):
            from ...ml_utils.metric_calculator import MetricCalculator
            self.metrics = MetricCalculator()
        self.metrics.reset()

        lam_slot, lam_sub = 0.0, 0.0
        if self._sara_alignment is not None:
            lam_slot, lam_sub = self._sara_alignment.get_lambdas(self._round_idx)

        loop = pbar(
            train_dl,
            desc=f"Adaptive SARA train (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
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

            align_loss = torch.tensor(0.0, device=self.device)
            if self._sara_alignment is not None and self._global_anchors is not None:
                unwrapped = self._unwrap(ta.model)
                param_map: Dict[str, nn.Parameter] = dict(unwrapped.named_parameters())
                for prefix in (self._lora_prefixes or []):
                    kA, kB = f"{prefix}.lora_A", f"{prefix}.lora_B"
                    if kA not in param_map or kB not in param_map:
                        continue
                    if kA not in self._global_anchors or kB not in self._global_anchors:
                        continue

                    A_loc = param_map[kA]
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
                    rank_weight = self._rank_adaptive_weight(layer_rank, global_rank)
                    align_loss = align_loss + rank_weight * (lam_slot * slot_loss + lam_sub * sub_loss)

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
            f"[Adaptive SARA Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={self.metrics.avg_loss:.6f} | batches={self.metrics.total_batch} | "
            f"samples={self.metrics.total_samples} | λ_s={lam_slot:.4f} λ_b={lam_sub:.4f}"
        )

        return self.metrics.get_stats()
