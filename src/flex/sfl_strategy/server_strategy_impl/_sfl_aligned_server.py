from __future__ import annotations

import copy
from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from flex.fed_strategy.strategy_args import StrategyArgs
from flex.ml_utils.model_utils import ModelUtils
from flex.sfl_strategy.server_strategy_impl._sfl_roundavg_server import SflRoundAvgServerStrategy


class _AlignedFullMlp(nn.Module):
    """
    Full MLP identical to NNModel_MnistNNBrenden, used only for evaluation.
    Layer names match so we can directly load FL global weights for comparison.

        Flatten → fc1(784→200) → ReLU
               → fc2(200→200) → ReLU
               → fc3(200→10)  → Softmax
    """

    def __init__(self) -> None:
        super().__init__()
        self._flatten = nn.Flatten()
        self._fc1 = nn.Linear(784, 200)
        self._fc2 = nn.Linear(200, 200)
        self._fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self._flatten(x)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        return F.softmax(self._fc3(x), dim=1)


class SflAlignedServerStrategy(SflRoundAvgServerStrategy):
    """
    SFL-Aligned server strategy.

    Identical to SflRoundAvgServerStrategy in every algorithmic detail
    (snapshot / reset / capture / weighted-average aggregation).

    What changes vs sfl_roundavg:
    - Uses sfl_aligned_client + sfl_aligned_server models whose layer names
      exactly match NNModel_MnistNNBrenden:
          client: _fc1            (784 → 200, ReLU)
          server: _fc2, _fc3      (200 → 200 → 10, ReLU + Softmax)
    - _compose_full_model_for_eval assembles an _AlignedFullMlp using
      those matching key names, so the stitched model is byte-for-byte
      equivalent to what FL produces.
    - initialize_aligned_state() helper splits a full FL model weight dict
      into client / server portions with zero overhead.
    """

    def __init__(self, args: StrategyArgs, server_node: Any) -> None:
        super().__init__(args, server_node)
        self._strategy_type = "sfl_aligned"

    # ------------------------------------------------------------------
    # Weight splitting utility
    # ------------------------------------------------------------------

    @staticmethod
    def split_fl_weights(full_state_dict: dict) -> tuple[dict, dict]:
        """
        Split a full NNModel_MnistNNBrenden state_dict into the client
        and server portions used by sfl_aligned.

        Client keys : _fc1.weight, _fc1.bias
        Server keys : _fc2.weight, _fc2.bias, _fc3.weight, _fc3.bias
        """
        client_keys = {"_fc1.weight", "_fc1.bias"}
        client_state = {k: v.detach().clone()
                        for k, v in full_state_dict.items() if k in client_keys}
        server_state = {k: v.detach().clone()
                        for k, v in full_state_dict.items() if k not in client_keys}
        return client_state, server_state

    # ------------------------------------------------------------------
    # Override: compose full model for evaluation
    # ------------------------------------------------------------------

    def _compose_full_model_for_eval(self, node_vars):
        """
        Stitch client front weights (_fc1) and server rear weights (_fc2, _fc3)
        into an _AlignedFullMlp for evaluation.

        Key names are identical to NNModel_MnistNNBrenden so the evaluator
        produces results directly comparable with FL.
        """
        client_front_list = getattr(node_vars, "client_front_weights", []) or []
        client_weight = client_front_list[0] if client_front_list else {}
        server_model = getattr(node_vars, "model", None)
        server_state = server_model.state_dict() if server_model is not None else {}

        full_model = _AlignedFullMlp()
        full_state = full_model.state_dict()

        # Client segment: _fc1
        for key in ("_fc1.weight", "_fc1.bias"):
            if key in client_weight:
                full_state[key] = client_weight[key].detach().clone()

        # Server segment: _fc2, _fc3
        for key in ("_fc2.weight", "_fc2.bias", "_fc3.weight", "_fc3.bias"):
            if key in server_state:
                full_state[key] = server_state[key].detach().clone()

        full_model.load_state_dict(full_state, strict=True)
        return full_model, full_state
