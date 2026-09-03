from typing import Optional, Callable
from torch.utils.data import DataLoader
from ml_data_loader import DatasetLoader, DatasetLoaderArgs, CustomDataset   # 按你的目录调整 import
from .noniid_distribution_generator import NoniidDistributionGenerator
from .noniid_data_generator import NoniidDataGenerator

class NoniidDataLoader(DatasetLoader):
    """
    Non-IID DatasetLoader, integrates NoniidDataGenerator with DatasetLoader interface.
    """

    def __init__(self):
        super().__init__()
        self._client_loaders: list[DataLoader] = []

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        Create client dataloaders based on args.
        """
        base_loader = DataLoader(
            args.dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers
        )

        generator = NoniidDataGenerator(base_loader)
        extra = args.extra or {}
        self._client_loaders = generator.generate_noniid_data(
            distribution=extra.get("distribution", "mnist_lt"),
            distribution_config=extra.get("distribution_config"),
            data_volum_list=extra.get("data_volum_list"),
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers
        )

        if self._client_loaders:
            self._data_loader = self._client_loaders[0]
        else:
            raise ValueError("No client loaders generated in NoniidDataLoader.")

    @property
    def client_loaders(self) -> list[DataLoader]:
        """返回所有 client 的 DataLoader"""
        return self._client_loaders
