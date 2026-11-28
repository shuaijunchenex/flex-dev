"""
Entry class of lora sample
"""

from flex.fed_node import FedNodeVars, FedNodeEventArgs
from flex.fed_runner import FedRunner
from flex.ml_utils import AppEntry, console
from flex.fl_algorithms.noniid.noniid_data_generator import NoniidDataGenerator
from flex.ml_data_loader.dataset_loader_factory import DatasetLoaderFactory
from flex.ml_data_loader.dataset_loader_args import DatasetLoaderArgs


class Entry(AppEntry):
    def __init__(self):
        super().__init__()    

        # Define runner, client, server, edge yaml variables, can be set outside manually
        self.runner_yaml = None
        self.client_yaml = None 
        self.edge_yaml = None 
        self.server_yaml = None 

    #override
    def run(self, training_rounds = 50, device = 'cpu'):

        # Yamls - if yamls are None, get yaml from app config file automatically
        if self.runner_yaml is None:
            self.runner_yaml = self.get_app_object("runner")
        if self.client_yaml is None:
            self.client_yaml = self.get_app_object("client_yaml")
        if self.edge_yaml is None:
            self.edge_yaml = self.get_app_object("edge_yaml")
        if self.server_yaml is None:
            self.server_yaml = self.get_app_object("server_yaml")

        # Training rounds
        self.fed_runner = FedRunner()       # Runner
        self.fed_runner.training_rounds = training_rounds
        self.fed_runner.with_yaml(self.runner_yaml)
        self.fed_runner.create_nodes()
        self.fed_runner.create_run_strategy()

        # Prepare server node and var
        server_var = FedNodeVars(self.server_yaml)
        server_var.set_device(device)
        server_var.prepare() #TODO: create server strategy
        self.__attach_event_handler(server_var)
        server_var.owner_nodes = self.fed_runner.server_node        # Two way binding
        server_var.prepare_strategy_only()
        self.fed_runner.server_node.node_var = server_var

        # Load data
        dataloader_args = DatasetLoaderArgs(self.server_yaml.get("data_loader", None))
        mnist_train_loader = DatasetLoaderFactory().create(dataloader_args)

        # NonIID handler
        allocated_noniid_data = NoniidDataGenerator(mnist_train_loader.data_loader).generate_noniid_data(distribution='mnist_lt_one_label')

        for i in range(len(allocated_noniid_data)):
            args = DatasetLoaderArgs({'name': 'custom', 
                                      'root': '../../../.dataset', 
                                      'split': '', 
                                      'batch_size': 64, 
                                      'shuffle': True, 
                                      'num_workers': 0, 
                                      'is_download': True, 
                                      'is_load_train_set': True, 
                                      'is_load_test_set': True,
                                      'dataset': allocated_noniid_data[i]})
            allocated_noniid_data[i] = DatasetLoaderFactory().create(args)

        # Prepare each client node and var
        client_var_list = []
        for index, node in enumerate(self.fed_runner.client_node_list):
            client_var = FedNodeVars(self.client_yaml)
            client_var.set_device(device)
            client_var.prepare() #TODO: create client strategy
            client_var.data_loader = allocated_noniid_data[index]
            client_var.data_sample_num = client_var.data_loader.data_sample_num
            client_var.trainer.set_train_loader(client_var.data_loader)
            self.__attach_event_handler(client_var)

            # Two way binding
            client_var.owner_nodes = node
            node.node_var = client_var
            client_var.prepare_strategy_only()
            client_var_list.append(client_var)

        self.fed_runner.run()
        
        return

    # Attach events to node variable object
    def __attach_event_handler(self, node_var):
        node_var.attach_event("on_prepare_data_loader", self.on_prepare_data_loader)
        node_var.attach_event("on_prepare_model", self.on_prepare_model)
        node_var.attach_event("on_prepare_loss_func", self.on_prepare_loss_func)
        node_var.attach_event("on_prepare_optimizer", self.on_prepare_optimizer)      #attach EVENT
        node_var.attach_event("on_prepare_strategy", self.on_prepare_strategy)    
        node_var.attach_event("on_prepare_extractor", self.on_prepare_extractor)
        node_var.attach_event("on_prepare_model", self.on_prepare_model)
        node_var.attach_event("on_prepare_loss_func", self.on_prepare_loss_func)
        node_var.attach_event("on_prepare_data_distribution", self.on_prepare_data_distribution)
        node_var.attach_event("on_prepare_data_handler", self.on_prepare_data_handler)
        node_var.attach_event("on_prepare_client_selection", self.on_prepare_client_selection)
        node_var.attach_event("on_prepare_trainer", self.on_prepare_trainer)
        node_var.attach_event("on_prepare_aggregation", self.on_prepare_aggregation)
        node_var.attach_event("on_prepare_extractor", self.on_prepare_extractor)
        node_var.attach_event("on_prepare_strategy", self.on_prepare_strategy)
        node_var.attach_event("on_prepare_training_logger", self.on_prepare_training_logger)        

    # Event handdlers
    # region
    def on_prepare_data_loader(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_loader event")
        return

    def on_prepare_model(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_model event")
        return

    def on_prepare_optimizer(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_optimizer event")
        return

    def on_prepare_loss_func(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_loss_func event")
        return

    def on_prepare_data_distribution(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_distribution event")
        return

    def on_prepare_data_handler(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_handler event")
        return

    def on_prepare_client_selection(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_client_selection event")
        return

    def on_prepare_trainer(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_training event")
        return

    def on_prepare_aggregation(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_aggregation event")
        return

    def on_prepare_strategy(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_strategy event")
        return

    def on_prepare_extractor(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_extractor event")
        return

    def on_prepare_training_logger(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_training_logger event")
        return
    # endregion
