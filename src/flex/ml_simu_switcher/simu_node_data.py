from __future__ import annotations
from typing import Any


class SimuNodeData:
    """
    Node data
    """
    
    def __init__(self, data: Any, from_id: str, to_id: str):
        """
        from node id
        """
        self.from_node_id: str = from_id

        """
        to node id
        """
        self.to_node_id: str = to_id

        """
        data
        """
        self.data: Any = data

        """
        data type
        """
        self._data_type = type(self.data)


    @property
    def data_type(self):
        return self._data_type


    def __str__(self):
        return "NodeData: " + self.from_node_id + "->" + self.to_node_id