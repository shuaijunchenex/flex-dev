from __future__ import annotations
from typing import Any

from ..ml_utils import EventArgs
from .simu_node import SimuNodeData

###
# Simulation node data event args
###

class SimuNodeDataEventArgs(EventArgs, SimuNodeData):
    def __init__(self, data: Any, from_id: str, to_id: str):
        EventArgs.__init__(self)
        SimuNodeData.__init__(self, data, from_id, to_id)

    def __str__(self):
        return "NodeEventArgs: " + self.from_node_id + "->" + self.to_node_id