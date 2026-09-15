from abc import ABC, abstractmethod
from aggregator.fedavg import FedAvgAggregator

class NodeManager(ABC):
    def __init__(self, node_list):
        """
        :param clients: List of client objects, each of which should have a run() method
                        that returns (updated_weights, train_record).
        """
        self.node_list = node_list

    @abstractmethod
    def train_round(self):
        """
        Abstract method: Coordinates training across multiple clients and 
        returns aggregated weights along with training records from each client.
        """
        pass

    @abstractmethod
    def add_nodes(self, node):
        self.node_list.append(node)

class LinearMultiClientRunner(NodeManager):
    def __init__(self, client_list):
        super().__init__(client_list)

    def train_round(self, participants):
        client_results = []
        for client in participants:
            # Assuming client.run() returns (updated_weights, train_record)
            train_record_pack = client.local_training()
            client_results.append(train_record_pack)

        return client_results
    
    def observe_round(self, participants):
        '''
        Collects observations from each client in the participants list.
        '''
        client_results = {}
        for client in participants:
            # Assuming client.run() returns (updated_weights, train_record)
            train_record_pack = client.observation()
            node_id = train_record_pack["node_id"]
            client_results[node_id] = train_record_pack

        return client_results

    def add_nodes(self, node):
        self.node_list.append(node)
