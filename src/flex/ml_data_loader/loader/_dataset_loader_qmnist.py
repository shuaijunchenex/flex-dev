from __future__ import annotations

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from torch.utils.data import DataLoader, ConcatDataset
'''
Dataset loader for Qmnist
'''
class DatasetLoader_Qmnist(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        # 默认 transform（与 MNIST 一致）：ToTensor -> Normalize -> Flatten
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

        self.task_type = "cv"

        # -------- 合并 QMNIST train (60k) + test50k (50k) --------
        train_set = datasets.QMNIST(
            root=args.root,
            what="train",
            transform=args.transform,
            download=args.is_download,
            compat=True,  # 与 MNIST 标签兼容
        )
        extra_set = datasets.QMNIST(
            root=args.root,
            what="test50k",
            transform=args.transform,
            download=args.is_download,
            compat=True,
        )

        # 拼接
        self._dataset = ConcatDataset([train_set, extra_set])
        self.data_sample_num = len(self._dataset)  # 应该是 110000

        # DataLoader for train
        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )
        # from collections import Counter

        # labels = [label for _, label in self._dataset]

        # counter = Counter(labels)

        # -------- 测试集（默认使用 test 10k） --------
        test_transform = getattr(args, "test_transform", None) or args.transform
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size

        test10k = datasets.QMNIST(
            root=args.root,
            what="test10k",
            transform=test_transform,
            download=args.is_download,
            compat=True,
        )
        test50k = datasets.QMNIST(
            root=args.root,
            what="test50k",
            transform=test_transform,
            download=args.is_download,
            compat=True,
        )

        # 拼接两个测试集
        self._test_dataset = ConcatDataset([test10k, test50k])

        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # self._test_dataset = datasets.QMNIST(
        #     root=args.root,
        #     what="test10k",
        #     transform=test_transform,
        #     download=args.is_download,
        #     compat=True,
        # )
        # self._test_data_loader = DataLoader(
        #     self._test_dataset,
        #     batch_size=test_batch_size,
        #     shuffle=False,
        #     num_workers=args.num_workers,
        # )
        return
