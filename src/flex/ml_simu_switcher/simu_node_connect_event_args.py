from __future__ import annotations

from ..ml_utils import EventArgs

###
# Simulation node connect event args
###

class SimuNodeConnectEventArgs(EventArgs):
    def __init__(self, which_node_id: str):
        super().__init__()
        self._which_node_id = which_node_id

    @property
    def which_node_id(self):
        return self._which_node_id