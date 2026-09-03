from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import console, ConfigLoader
from flex.ml_data_loader import DatasetLoaderFactory
from flex.ml_data_process import DataHandler_Noniid, DataHandlerArgs


def main():
    yaml_file_name = './test_data/node_config_template_client.yaml'

    # Form yaml file
    console.out(f"Test from yaml file: {yaml_file_name}")
    console.out("------------- Begin ---------------")

    yaml = ConfigLoader.load(yaml_file_name)
    console.out(yaml)

    args = DatasetLoaderFactory.create_args(yaml)
    args.root = "../../../.dataset"

    print(args)

    dataset_loader = DatasetLoaderFactory.create(args)
    print(dataset_loader)

    data_handler = DataHandler_Noniid(dataset_loader.data_loader)
    handler_args = DataHandlerArgs()
    handler_args.batch_size = args.batch_size
    handler_args.num_workers = args.num_workers
    loaders = data_handler.generate(handler_args)

    for l in loaders:
        console.info(f"{l}")

    console.out("------------- End -----------------")
    return


if __name__ == "__main__":
    main()

