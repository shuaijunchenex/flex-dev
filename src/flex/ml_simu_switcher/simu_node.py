from __future__ import annotations
from queue import Queue
from typing import Any

from .simu_node_connect_event_args import SimuNodeConnectEventArgs
from .simu_node_data import SimuNodeData
from ..ml_utils import EventHandler


class SimuNode(EventHandler):
    """
    Simulate Node
    Events:
       on_receive : receive data from another node
       on_connected : raise when connection established
       on_disconnected : raise when connection close
    """
    def __init__(self, switcher):
        super().__init__()
        self._switcher = switcher

        # local node id
        self._local_node_id: str

        # remote node id after connected
        self._remote_node_id: str

        # node is connected to another node
        self._is_connected: bool = False

        # receive data queue
        self._queue: Queue = Queue()

        # node clients array
        self._clients: list = []

        self.declare_events("on_connected", "on_disconnected", "on_receive")
        return

    ############################################################
    # region Attributes

    @property
    def remote_node_id(self):
        return self._remote_node_id

    @property
    def local_node_id(self):
        return self._local_node_id
    # endregion

    ############################################################

    def connect(self, remote_node_id: str):
        """
        Connect node to a remote node(id)
        """
        if not self._switcher.node_exists(remote_node_id):
            raise Exception("Remote node not exists")

        if self._is_connected:      #node has connected to node
            return

        self._switcher.link(self._local_node_id, remote_node_id)
        self._remote_node_id = remote_node_id
        self._is_connected = True
        self.raise_event("on_connected", None)
        return

    def close(self):
        """
        Close connection
        """
        if not self._is_connected:
            return

        #Close all connection
        for node_id in self._clients:
            self._switcher.unlink(self._local_node_id, node_id)
        
        self._clients.clear()
        self._is_connected = False
        return


    def accept(self, node_id: str):
        """
        Accept a client connection
        """
        if node_id not in self._clients:
            self._clients.append(node_id)

            args = SimuNodeConnectEventArgs(node_id).with_sender(self)
            self.raise_event("on_connected", args)      #raise connected event
        return


    def send_up(self, data: Any):
        """
        Send data to connected up server
        """
        self.send_to(self._remote_node_id, data)
        return


    def send_down(self, node_id: str, data: Any):
        """
        Send data to a client which connected to this node
        """
        if node_id not in self._clients:
            raise Exception("Client node(" + node_id + ") not connected")

        self.send_to(node_id, data)
        return


    def send_down_all(self, data: Any):
        """
        Send data to all clients connected to this node
        """
        for id in self._clients:
            self.send_to(id, data)
        return


    def send_to(self, to_node_id: str, data: Any):
        """
        Send data to node id
        """
        if not self._switcher.link_exists(self._local_node_id, to_node_id):
            raise Exception("Node not connect to remove node(id: " + to_node_id + ")")

        node_data = SimuNodeData(data, self._local_node_id, to_node_id)
        self._switcher.send_node_data(node_data)
        return
