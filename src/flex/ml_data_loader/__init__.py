# Package

from .dataset_loader import DatasetLoader
from .dataset_loader_args import DatasetLoaderArgs
from .dataset_loader_util import DatasetLoaderUtil
from .dataset_loader_factory import DatasetLoaderFactory
from .dataset_custom import CustomDataset

# from .loader._dataset_loader_minst import DatasetLoader_Mnist
# from .loader._dataset_loader_fminst import DatasetLoader_Fmnist
# from .loader._dataset_loader_cifar10 import DatasetLoader_Cifar10
# from .loader._dataset_loader_cifar100 import DatasetLoader_Cifar100
# from .loader._dataset_loader_emnist import DatasetLoader_Emnist
# from .loader._dataset_loader_kmnist import DatasetLoader_Kmnist
# from .loader._dataset_loader_qmnist import DatasetLoader_Qmnist
# from .loader._dataset_loader_stl10 import DatasetLoader_Stl10
# from .loader._dataset_loader_svhn import DatasetLoader_Svhn
# from .loader._dataset_loader_imagenet import DatasetLoader_ImageNet
# from .loader._dataset_loader_agnews import DatasetLoader_Agnews
# from .loader._dataset_loader_imdb import DatasetLoader_Imdb
# from .loader._dataset_loader_dbpedia import DatasetLoader_DBpedia
# from .loader._dataset_loader_yahooanswers import DatasetLoader_YahooAnswers

__all__ = [
    "DatasetLoader",
    "DatasetLoaderArgs",
    "DatasetLoaderFactory",
    "DatasetLoaderUtil",
    "CustomDataset",
    # "DatasetLoader_Mnist",
    # "DatasetLoader_Fmnist",
    # "DatasetLoader_Cifar10",
    # "DatasetLoader_Cifar100",
    # "DatasetLoader_Emnist",
    # "DatasetLoader_Kmnist",
    # "DatasetLoader_Qmnist",
    # "DatasetLoader_Stl10",
    # "DatasetLoader_Svhn",
    # "DatasetLoader_ImageNet",
    # "DatasetLoader_Agnews",
    # "DatasetLoader_Imdb",
    # "DatasetLoader_DBpedia",
    # "DatasetLoader_YahooAnswers"
]