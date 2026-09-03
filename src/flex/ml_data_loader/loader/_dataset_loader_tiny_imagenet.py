from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

'''
Dataset loader for Tiny-ImageNet (200 classes, 64×64 RGB).
Uses HuggingFace `datasets` library (zh-plus/tiny-imagenet).
  - train: 100 000 samples (500 × 200 classes)
  - valid: 10 000 samples  (50  × 200 classes)

No manual download or directory reorganisation needed.
Set is_download: True in yaml to enable auto-download (cached under <root>/hf_datasets/).

Mean / std (per-channel, computed on the training set):
    mean = (0.4802, 0.4481, 0.3975)
    std  = (0.2302, 0.2265, 0.2262)
'''

_HF_REPO_ID = "zh-plus/tiny-imagenet"

_TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
_TINYIMAGENET_STD  = (0.2302, 0.2265, 0.2262)

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(_TINYIMAGENET_MEAN, _TINYIMAGENET_STD),
])


class _HFTinyImageNetDataset(Dataset):
    """Wraps a HuggingFace Tiny-ImageNet split and applies a torchvision transform."""

    def __init__(self, hf_split, transform):
        self._data = hf_split
        self._transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data[idx]
        img   = sample["image"].convert("RGB")   # PIL Image
        label = int(sample["label"])
        if self._transform is not None:
            img = self._transform(img)
        return img, label


class DatasetLoader_TinyImageNet(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        from datasets import load_dataset  # type: ignore

        cache_dir       = os.path.join(args.root, "hf_datasets")
        batch_size      = args.batch_size
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        num_workers     = args.num_workers
        shuffle         = args.shuffle

        transform      = args.transform or _DEFAULT_TRANSFORM
        test_transform = getattr(args, "test_transform", None) or transform

        # Load both splits at once (cached locally after first download)
        ds = load_dataset(_HF_REPO_ID, cache_dir=cache_dir)

        self._dataset      = _HFTinyImageNetDataset(ds["train"], transform)
        self._test_dataset = _HFTinyImageNetDataset(ds["valid"], test_transform)

        self.data_sample_num = len(self._dataset)   # 100 000
        self.task_type = "cv"

        generator = self.make_generator(args)

        self._data_loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
        )

        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return
