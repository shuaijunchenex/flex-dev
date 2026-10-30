from __future__ import annotations

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

'''
Dataset loader for svhn
'''
class DatasetLoader_Svhn(DatasetLoader):
    def __init__(self):
        super().__init__()

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.task_type = "cv"

        self._dataset = datasets.SVHN(root=args.root, split=args.split, transform=args.transform, download=args.is_download)
        self.data_sample_num = len(self._dataset)  # train=73257, test=26032, extra=531131
        generator = self.make_generator(args)
        self._data_loader = DataLoader(self._dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, generator=generator)

        test_transform = getattr(args, "test_transform", None) or args.transform
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size

        self._test_dataset = datasets.SVHN(root=args.root, split="test", transform=test_transform, download=args.is_download)
        self._test_data_loader = DataLoader(self._test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers)
        return
