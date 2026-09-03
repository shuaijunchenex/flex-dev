from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for Cifar100
'''
class DatasetLoader_Cifar100(DatasetLoader):
    def __init__(self):
        super().__init__()

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),                
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])      # Standard CIFAR100 normalization values

        self.data_sample_num = 50000
        self.task_type = "cv"

        self._dataset = datasets.CIFAR100(root=args.root, train=args.is_train, transform=args.transform, download=args.is_download)
        generator = self.make_generator(args)
        self._data_loader = DataLoader(self._dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, generator=generator)

        test_transform = getattr(args, "test_transform", None) or args.transform
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size

        self._test_dataset = datasets.CIFAR100(root=args.root, train=False, transform=test_transform, download=args.is_download)
        self._test_data_loader = DataLoader(self._test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers)
        return
