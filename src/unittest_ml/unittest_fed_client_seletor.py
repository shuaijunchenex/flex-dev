from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.fl_algorithms import FedClientSelectorFactory, FedClientSelectorArgs
from flex.ml_utils import console, ConfigLoader


##############################################

simu_clients = [i for i in range(10)]
simu_clients_data = {}
for i in range(10):
    simu_clients_data[i] = i

yaml_file_name = './test_data/node_config_template_server.yaml'

# Load yaml file
console.out(f"Test client selection from {yaml_file_name}")
console.out(f"load yaml file: {yaml_file_name}")
yaml = ConfigLoader.load(yaml_file_name)

def test_client_selector(method):
    # Create selector args
    # - Direct create from class
    # args = FedClientSelectorArgs(yaml)
    # or create from factory
    args = FedClientSelectorFactory.create_args(yaml)
    args.select_method = method

    # Create selector from factory
    selector = FedClientSelectorFactory.create(args)   #create selector    

    console.info("Select client by args number:")
    selected_clients = selector.select(simu_clients)            #select client
    print(selected_clients)

    console.info("Select client by specified number:")
    selected_clients = selector.select(simu_clients, 5)            #select client
    print(selected_clients)
    return


def test_client_selector_high_loss():
   # Create selector args
    args = FedClientSelectorFactory.create_args(yaml)
    args.select_method = "high_loss"        
    selector = FedClientSelectorFactory.create(args).with_clients_data(simu_clients_data)

    selected_client = selector.select(simu_clients, 5)            #select client
    print(selected_client)
    return

def main():
    test_client_selector("random")
    test_client_selector("all")

    # test_client_selector_high_loss()  
    return

if __name__ == "__main__":
    main()
