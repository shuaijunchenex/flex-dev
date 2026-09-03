from __future__ import annotations

import copy
from typing import Any, Tuple

import torch.nn as nn

from flex.fed_strategy.strategy_args import StrategyArgs
from flex.fed_strategy.client_strategy_impl._fedavg_client import FedAvgClientTrainingStrategy
from flex.ml_utils.model_utils import ModelUtils
from flex.ml_utils import console
from flex.fed_node.fed_node_vars import FedNodeVars


# ---------------------------------------------------------------------------
# Metrics that must be present in train_record for all supported selectors
# ---------------------------------------------------------------------------
_REQUIRED_METRICS = (
    "node_id",                        # selector routing key
    "data_sample_num",                # weighted aggregation
    "avg_loss",                       # general monitoring
    "sqrt_train_loss_power_two_sum",  # high_loss / low_loss selector
    "weight_l2_before",               # weight norm before training
    "weight_l2_after",                # weight norm after training
    "weight_l2_delta",                # high_weight_divergence selector  ‖w_local − w_global‖₂
)


def _enrich_train_record(train_record: dict, node_id: Any, data_sample_num: Any) -> dict:
    """
    Inject ``node_id`` and ``data_sample_num`` directly into *train_record*
    so the dict is self-contained and the server-side feedback parser does not
    need to fall back to the outer wrapper.

    Also logs a warning for any required metric that is missing so
    misconfigurations surface at training time rather than silently producing
    wrong selector scores.
    """
    enriched = dict(train_record)
    enriched["node_id"] = node_id
    enriched["data_sample_num"] = data_sample_num

    missing = [k for k in _REQUIRED_METRICS if enriched.get(k) is None]
    if missing:
        console.warn(
            f"[ClientSelectionPurpose] client {node_id}: "
            f"train_record is missing metrics {missing}. "
            f"Selector scores may be incorrect."
        )
    else:
        console.debug(
            f"[ClientSelectionPurpose] client {node_id}: "
            f"avg_loss={enriched.get('avg_loss', 'N/A'):.4f}  "
            f"weight_l2_delta={enriched.get('weight_l2_delta', 'N/A'):.4f}"
        )
    return enriched


class ClientSelectionPurposeClientStrategy(FedAvgClientTrainingStrategy):
    """
    Client strategy designed for use with ``ClientSelectionPurposeServerStrategy``.

    Identical training logic to ``FedAvgClientTrainingStrategy`` but enriches
    every returned ``train_record`` with the metrics required by all supported
    selectors:

    +---------------------------------+-------------------------------------+
    | Metric                          | Used by selector                    |
    +=================================+=====================================+
    | ``node_id``                     | All (routing key for feedback dict) |
    +---------------------------------+-------------------------------------+
    | ``data_sample_num``             | Weighted FedAvg aggregation         |
    +---------------------------------+-------------------------------------+
    | ``avg_loss``                    | General monitoring / logging        |
    +---------------------------------+-------------------------------------+
    | ``sqrt_train_loss_power_two_sum``| ``high_loss``, ``low_loss``        |
    +---------------------------------+-------------------------------------+
    | ``weight_l2_before``            | General monitoring / logging        |
    +---------------------------------+-------------------------------------+
    | ``weight_l2_after``             | General monitoring / logging        |
    +---------------------------------+-------------------------------------+
    | ``weight_l2_delta``             | ``high_weight_divergence``          |
    +---------------------------------+-------------------------------------+

    Trainer check
    -------------
    ``ModelTrainer_Standard.train()`` already computes and returns all weight /
    loss metrics listed above inside ``train_stats``.  This strategy only adds
    ``node_id`` and ``data_sample_num``, which the trainer cannot know.
    If any expected metric is absent (e.g. a custom trainer omits it) a warning
    is emitted at runtime so misconfigurations are detected early.
    """

    def __init__(self, args: StrategyArgs, client_node: Any) -> None:
        super().__init__(args, client_node)
        self._strategy_type = "client_selection_purpose"

    # ------------------------------------------------------------------
    # Override: run_local_training  — real training, write-back to node_var
    # ------------------------------------------------------------------
    def run_local_training(self) -> Tuple[dict, dict]:
        updated_weights, raw_train_record = self.local_training_step()

        train_record = _enrich_train_record(
            raw_train_record,
            node_id=self._obj.node_id,
            data_sample_num=self._obj.node_var.data_sample_num,
        )

        return updated_weights, {
            "node_id":          self._obj.node_id,
            "updated_weights":  updated_weights,
            "train_record":     train_record,
            "data_sample_num":  self._obj.node_var.data_sample_num,
        }

    # ------------------------------------------------------------------
    # Override: observation_step — no write-back, metrics only
    # ------------------------------------------------------------------
    def observation_step(self) -> Tuple[dict, dict]:
        updated_weights, raw_train_record = super().observation_step()

        train_record = _enrich_train_record(
            raw_train_record,
            node_id=self._obj.node_id,
            data_sample_num=self._obj.node_var.data_sample_num,
        )
        return updated_weights, train_record

    # ------------------------------------------------------------------
    # Override: run_observation  — public wrapper for observer runner
    # ------------------------------------------------------------------
    def run_observation(self) -> dict:
        console.info(f"\n Observation Client [{self._obj.node_id}] ...\n")
        updated_weights, train_record = self.observation_step()
        return {
            "node_id":         self._obj.node_id,
            "train_record":    train_record,      # already enriched
            "data_sample_num": self._obj.node_var.data_sample_num,
        }
