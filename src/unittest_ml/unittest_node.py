
# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import ConfigLoader
from flex.fed_node import FedNodeClient, FedNodeServer, FedNodeEventArgs

def load_config():
    cfg_filename = './test_data/node_config_template_client.yaml'
    config_dict = ConfigLoader.load(cfg_filename)
    return config_dict

def test_node_client():
    node_config = load_config()
    print(f"Node config:\n{node_config}")

    # Create client node with args
    client_node = FedNodeClient("client.1", node_config)

    client_node.attach_event("on_prepare_optimizer", on_prepare_optimizer)      #attach EVENT
    client_node.attach_event("on_prepare_strategy", on_prepare_strategy)    
    client_node.attach_event("on_prepare_extractor", on_prepare_extractor)
    client_node.attach_event("on_prepare_dataset", on_prepare_dataset)
    client_node.attach_event("on_prepare_model", on_prepare_model)
    client_node.attach_event("on_prepare_loss_func", on_prepare_loss_func)

    #client_node.run()
    return

#-------------------------
# Event callbacks
#-------------------------
def on_prepare_dataset(event_args: FedNodeEventArgs):
    # Will call after dataset is built
    print(f"Arg kind: {event_args.kind}")
    print(f"Args: {event_args.client_args}")
    print(f"on_build_dateset event occur {event_args}")
    return

def on_prepare_model(event_args: FedNodeEventArgs):
    # Will call after model is built
    print(f"Arg kind: {event_args.kind}")
    print(f"on_build_model event occur {event_args}")
    return

def on_prepare_loss_func(event_args: FedNodeEventArgs):
    # Call to build extractor
    print(f"Arg kind: {event_args.kind}")
    print(f"on_build_loss_func event occur {event_args}")
    return

def on_prepare_optimizer(event_args: FedNodeEventArgs):
    # Call to build optimizer
    print(f"Arg kind: {event_args.kind}")
    print(f"on_build_optimizer event occur {event_args}")
    return

def on_prepare_strategy(event_args: FedNodeEventArgs):
    # Call to build strategy
    print(f"Arg kind: {event_args.kind}")
    print(f"on_build_strategy event occur {event_args}")
    return

def on_prepare_extractor(event_args: FedNodeEventArgs):
    # Call to build extractor
    print(f"Arg kind: {event_args.kind}")
    print(f"on_build_extractor event occur {event_args}")
    return


def test_node_server():
    yaml_config = load_config()
    print("Loaded YAML data:")
    print(yaml_config)

    server_node = FedNodeServer(yaml_config)
    server_node.run()
    return


def main():
    test_node_client()
    test_node_server()


if __name__ == "__main__":
    main()
