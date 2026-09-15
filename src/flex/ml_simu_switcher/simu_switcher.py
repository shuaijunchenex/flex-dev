from __future__ import annotations
from queue import Queue
from threading import Event, Thread

from .simu_node import SimuNode, SimuNodeData
from .simu_node_data_event_args import SimuNodeDataEventArgs
from .simu_node_disconnect_event_args import SimuNodeDisconnectEventArgs


class SimuSwitcher:
    def __init__(self):
        self._thread = Thread(target=self._run_loop)  # Switcher thread
        self._is_running: bool = False          # indicate thread is running
        self._stop_event = Event()
        self._wait_event = Event()
        self._send_queue = Queue()

        self._node_dict = {}
        self._link_array = []
        return

    @property
    def is_running(self):
        return self._is_running

    def create_node(self, node_id: str) -> SimuNode:
        """
        Create a Node
        """
        if node_id is None or len(node_id) == 0:
            raise Exception("Node ID is empty")

        if node_id in self._node_dict:
            raise Exception(f"Node(id: {node_id}) alreay exists, unable to create again")

        node = SimuNode(self)
        node._local_node_id = node_id

        self._node_dict[node_id] = node
        return node


    def link(self, from_node_id, to_node_id):
        """
        Make link bwtween two node, via connection
        """
        if self.link_exists(from_node_id, to_node_id):
            return

        to_node: SimuNode = self._node_dict[to_node_id]
        to_node.accept(from_node_id)

        self._link_array.append(self.__link_id(from_node_id, to_node_id))
        return


    def unlink(self, from_node_id, to_node_id):
        """
        Unlink bwtween two node, via disconnect
        """
        if not self.link_exists(from_node_id, to_node_id):
            return

        to_node: SimuNode = self._node_dict[to_node_id]
        self._link_array.remove(self.__link_id(from_node_id, to_node_id))
        
        args = SimuNodeDisconnectEventArgs(to_node_id).with_sender(to_node)
        to_node.raise_event("on_disconnected", args)


    def link_exists(self, node_id_1, node_id_2):
        """
        check connection link exists
        """
        id1 = self.__link_id(node_id_1, node_id_2)
        id2 = self.__link_id(node_id_2, node_id_1)
        return id1 in self._link_array or id2 in self._link_array


    def send_node_data(self, data: SimuNodeData):
        """
        Send node data
        """
        self._send_queue.put(data)
        self._wait_event.set()
        return


    def node_exists(self, node_id: str) -> bool:
        """
        Check if node has created
        """
        return node_id in self._node_dict


    def __link_id(self, from_node_id, to_node_id):
        """
        Private method to generate link id
        """
        return f"{from_node_id}<->{to_node_id}"

    ##################################################
    # thread

    def run(self):
        """
        Run switcher(thread)
        """
        if self._is_running:
            return

        self._stop_event.clear()
        self._wait_event.clear()

        self._thread.start()
        self._is_running = True
        return


    def stop(self):
        """
        Stop switcher(thread)
        """
        if not self._is_running:
            return

        self._stop_event.set()
        self._wait_event.set()
        self._thread.join()

        self._is_running = False
        return


    def _run_loop(self):
        """
        " Thread loop
        """
        while not self._stop_event.is_set():
            if self._send_queue.qsize() <= 0:
                self._wait_event.clear()
                self._wait_event.wait()  # wait until event set
                continue

            # node data from queue
            node_data: SimuNodeData = self._send_queue.get()

            # notify event
            eventArgs = SimuNodeDataEventArgs(node_data.data, node_data.from_node_id, node_data.to_node_id)
            node: SimuNode = self._node_dict[node_data.to_node_id]      #which node
            node.raise_event("on_receive", eventArgs)
        return
