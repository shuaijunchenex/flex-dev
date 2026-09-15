from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for minst
'''
class DatasetLoader_Fmnist(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        # 默认 transform
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

        self.data_sample_num = 60000
        self.task_type = "cv"

        # 训练集
        self._dataset = datasets.FashionMNIST(
            root=args.root,
            train=args.is_train,
            transform=args.transform,
            download=args.is_download
        )
        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers
        )

        # 测试集（可选 test_transform / test_batch_size）
        test_transform = getattr(args, "test_transform", None) or args.transform
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size

        self._test_dataset = datasets.FashionMNIST(
            root=args.root,
            train=False,
            transform=test_transform,
            download=args.is_download
        )
        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        return
