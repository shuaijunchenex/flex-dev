from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Callable, Union
from ..ml_utils.key_value_args import KeyValueArgs

Object = str  # "client" | "server"
StrategyOwner = Union[object]  # 这里 owner 既可能是 client 也可能是 server/fednode

@dataclass
class StrategyArgs(KeyValueArgs):

    role: str = None                     
    strategy_name: str = None
    preserve_optimizer_state: bool = False   # When True, optimizer momentum/state persists across rounds
    offload_weights_to_cpu: bool = True      # When True, training results are moved to CPU to free GPU memory

    def __init__(self, config_dict: Optional[dict] = None, is_clone_dict: bool = False):
        KeyValueArgs.__init__(self, config_dict, is_clone_dict)
        self.role = self.get("role", self.role).lower()
        if self.role not in ("client", "server", "runner"):
            raise ValueError(f"[StrategyArgs] 'role' must be 'client', 'server' or 'runner', got: {self.role}")
        self.strategy_name = self.get("strategy_name", self.strategy_name)
        self.preserve_optimizer_state = bool(self.get("preserve_optimizer_state", False))
        self.offload_weights_to_cpu = bool(self.get("offload_weights_to_cpu", True))