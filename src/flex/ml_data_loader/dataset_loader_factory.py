from __future__ import annotations
from typing import Callable

from torch.utils.data import DataLoader

from .dataset_loader import DatasetLoader
from .dataset_loader_args import DatasetLoaderArgs

class DatasetLoaderFactory:
    """
    " Dataset loader factory
    """

    @staticmethod
    def create_loader(dataset, batch_size, shuffle=True, num_workers=4, collate_fn=None):
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset: The dataset to load data from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            num_workers (int): Number of worker threads.
            collate_fn (callable, optional): Function to merge a list of samples into a batch.

        Returns:
            DataLoader: A PyTorch DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn)

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> DatasetLoaderArgs:
        """
        " Static method to create data loader args
        """
        return DatasetLoaderArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(data_loader_args: DatasetLoaderArgs, fn:Callable[[DatasetLoader]]|None = None) -> DatasetLoader:
        """
        " Static method to create data loader
        """
        match data_loader_args.dataset_type.lower():
            case "mnist":
                from .loader._dataset_loader_mnist import DatasetLoader_Mnist
                return DatasetLoader_Mnist().create(data_loader_args, fn)
            case "fmnist":
                from .loader._dataset_loader_fmnist import DatasetLoader_Fmnist
                return DatasetLoader_Fmnist().create(data_loader_args, fn)
            case "cifar10":
                from .loader._dataset_loader_cifar10 import DatasetLoader_Cifar10
                return DatasetLoader_Cifar10().create(data_loader_args, fn)
            case "cifar100":
                from .loader._dataset_loader_cifar100 import DatasetLoader_Cifar100
                return DatasetLoader_Cifar100().create(data_loader_args, fn)
            case "emnist":
                from .loader._dataset_loader_emnist import DatasetLoader_Emnist
                return DatasetLoader_Emnist().create(data_loader_args, fn)
            case "kmnist":
                from .loader._dataset_loader_kmnist import DatasetLoader_Kmnist
                return DatasetLoader_Kmnist().create(data_loader_args, fn)
            case "qmnist":
                from .loader._dataset_loader_qmnist import DatasetLoader_Qmnist
                return DatasetLoader_Qmnist().create(data_loader_args, fn)
            case "stl10":
                from .loader._dataset_loader_stl10 import DatasetLoader_Stl10
                return DatasetLoader_Stl10().create(data_loader_args, fn)
            case "svhn":
                from .loader._dataset_loader_svhn import DatasetLoader_Svhn
                return DatasetLoader_Svhn().create(data_loader_args, fn)
            case "imagenet":
                from .loader._dataset_loader_imagenet import DatasetLoader_ImageNet
                return DatasetLoader_ImageNet().create(data_loader_args, fn)
            case "agnews":
                from .loader._dataset_loader_agnews import DatasetLoader_Agnews
                return DatasetLoader_Agnews().create(data_loader_args, fn)
            case "imdb":
                from .loader._dataset_loader_imdb import DatasetLoader_Imdb
                return DatasetLoader_Imdb().create(data_loader_args, fn)
            case "sst2":
                from .loader._dataset_loader_sst2 import DatasetLoader_SST2
                return DatasetLoader_SST2().create(data_loader_args, fn)
            case "mnli":
                from .loader._dataset_loader_mnli import DatasetLoader_MNLI
                return DatasetLoader_MNLI().create(data_loader_args, fn)
            case "qnli":
                from .loader._dataset_loader_qnli import DatasetLoader_QNLI
                return DatasetLoader_QNLI().create(data_loader_args, fn)
            case "rte":
                from .loader._dataset_loader_rte import DatasetLoader_RTE
                return DatasetLoader_RTE().create(data_loader_args, fn)
            case "wnli":
                from .loader._dataset_loader_wnli import DatasetLoader_WNLI
                return DatasetLoader_WNLI().create(data_loader_args, fn)
            case "qqp":
                from .loader._dataset_loader_qqp import DatasetLoader_QQP
                return DatasetLoader_QQP().create(data_loader_args, fn)
            case "cola":
                from .loader._dataset_loader_cola import DatasetLoader_CoLA
                return DatasetLoader_CoLA().create(data_loader_args, fn)
            case "mrpc":
                from .loader._dataset_loader_mrpc import DatasetLoader_MRPC
                return DatasetLoader_MRPC().create(data_loader_args, fn)
            case "dbpedia":
                from .loader._dataset_loader_dbpedia import DatasetLoader_DBpedia
                return DatasetLoader_DBpedia().create(data_loader_args, fn)
            case "yahooanswers":
                from .loader._dataset_loader_yahooanswers import DatasetLoader_YahooAnswers
                return DatasetLoader_YahooAnswers().create(data_loader_args, fn)
            case "custom":
                from .loader._dataset_loader_custom import DatasetLoader_Custom
                return DatasetLoader_Custom().create(data_loader_args, fn)
            case "cinic10":
                from .loader._dataset_loader_cinic10 import DatasetLoader_Cinic10
                return DatasetLoader_Cinic10().create(data_loader_args, fn)
            case "tiny_imagenet":
                from .loader._dataset_loader_tiny_imagenet import DatasetLoader_TinyImageNet
                return DatasetLoader_TinyImageNet().create(data_loader_args, fn)

        raise ValueError(f"Datasdet type '{data_loader_args.dataset_type}' not support.")
