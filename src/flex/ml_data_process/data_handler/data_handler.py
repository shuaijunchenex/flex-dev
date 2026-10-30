from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from .data_handler_args import DataHandlerArgs

"""
Abstract Data Handler
"""

class DataHandler(ABC):
    def __init__(self, dataloader):
        """
        Args:
            dataloader (DataLoader): PyTorch DataLoader for dataset
        """

        self.dataloader = dataloader
        self.data_pool = None
        self.x_train = []
        self.y_train = []
        self.train_loaders = []

        # Load data into memory
        self._load_data()
        self.create_data_pool()
        return

    @abstractmethod
    def _load_data(self):
        """
        Load data from DataLoader and store in x_train, y_train
        """
        pass

    @abstractmethod
    def create_data_pool(self, pools = 10) -> Any:
        """
        Organizes dataset into a dictionary where keys are class labels (0-9),
        and values are lists of corresponding images.

        Returns:
            dict: {label: tensor(images)}
        """
        return self.data_pool

    @abstractmethod
    def generate(self, args: DataHandlerArgs):
        """
        Distributes imbalanced data to different clients based on predefined patterns and returns a list of DataLoader for each client.

        Args:
            data_volum_list (list): A list containing data volume for different classes (used only if distribution="custom").
            verify_allocate (bool): Whether to print allocation results.
            distribution (str): Default is "mnist_lt", supports different distributions.
            batch_size (int): Number of samples per batch for the DataLoader.
            shuffle (bool): Whether to shuffle the data in the DataLoader.
            num_workers (int): Number of worker threads for the DataLoader.

        Returns:
            list: A list of DataLoader objects, each corresponding to one client's data.
        """

        return self.train_loaders
