from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# CIFAR-10 normalization constants (per-channel mean & std)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# FedAvg-style training augmentation (ref: McMahan et al., 2017)
#   32×32 → RandomCrop(24) → RandomHorizontalFlip →
#   ColorJitter(brightness, contrast) → ToTensor → Normalize
FEDAVG_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(24),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# FedAvg-style test transform
#   32×32 → CenterCrop(24) → ToTensor → Normalize
FEDAVG_TEST_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(24),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

'''
Dataset loader for Cifar10
'''
class DatasetLoader_Cifar10(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        # Use FedAvg-style augmentation by default; allow override via args.transform
        if args.transform is None:
            args.transform = FEDAVG_TRAIN_TRANSFORM

        self.data_sample_num = 50000
        self.task_type = "cv"

        self._dataset = datasets.CIFAR10(
            root=args.root, train=args.is_train,
            transform=args.transform, download=args.is_download
        )
        generator = self.make_generator(args)
        self._data_loader = DataLoader(
            self._dataset, batch_size=args.batch_size,
            shuffle=args.shuffle, num_workers=args.num_workers,
            generator=generator,
        )

        # Test transform: use explicit test_transform if provided, else FedAvg default
        test_transform = getattr(args, "test_transform", None)
        if test_transform is None:
            test_transform = FEDAVG_TEST_TRANSFORM
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size

        self._test_dataset = datasets.CIFAR10(
            root=args.root, train=False,
            transform=test_transform, download=args.is_download
        )
        self._test_data_loader = DataLoader(
            self._test_dataset, batch_size=test_batch_size,
            shuffle=False, num_workers=args.num_workers
        )
        return

