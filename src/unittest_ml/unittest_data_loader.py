from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import console, ConfigLoader
from flex.ml_data_loader import DatasetLoaderFactory, DatasetLoader

def after_create_fn(loader: DatasetLoader):
    console.info(f"\nAfter create fn object: {loader}")
    return

def test_dataset_loader(args, dataset_type):
    args.dataset_type = dataset_type
    args.root = "../../../.dataset"
    args.is_train = True
    args.is_download = True

    dataset_loader = DatasetLoaderFactory.create(args, after_create_fn)
    print(dataset_loader)
    return


def main():
    yaml_file_name = './test_data/node_config_template_client.yaml'

    # Form yaml file
    console.out(f"Test from yaml file: {yaml_file_name}")
    console.out("------------- Begin ---------------")

    yaml = ConfigLoader.load(yaml_file_name)
    # console.out(yaml)

    args = DatasetLoaderFactory.create_args(yaml)
    
    test_dataset_loader(args, "mnist")
    test_dataset_loader(args, "fmnist")

    args.split = 'balanced'
    test_dataset_loader(args, "emnist")

    test_dataset_loader(args, "kmnist")
    test_dataset_loader(args, "qmnist")

    test_dataset_loader(args, "cifar10")
    test_dataset_loader(args, "cifar100")

    args.split = 'train'
    test_dataset_loader(args, "stl10")

    # args.split = 'train'
    # test_dataset_loader(args, "imagenet")

    console.out("------------- End -----------------")
    return


if __name__ == "__main__":
    main()

