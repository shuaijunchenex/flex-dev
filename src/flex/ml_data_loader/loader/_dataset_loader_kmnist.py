from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from torch.utils.data import DataLoader
# import os
# import time
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.datasets.utils import check_integrity, extract_archive
# import torch
from ._hf_dataset import load_hf_dataset
from ._hf_image_dataset import HFImageDataset

'''
Dataset loader for Kmnist
'''

# Previous mirror patch (the GitHub path does not contain these data files):
# _KMNIST_GITHUB_MIRROR = "https://github.com/rois-codh/kmnist/raw/master/dataset/kmnist/"
# if _KMNIST_GITHUB_MIRROR not in datasets.KMNIST.mirrors:
#     datasets.KMNIST.mirrors = [_KMNIST_GITHUB_MIRROR] + list(datasets.KMNIST.mirrors)


# Previous torchvision archive loading retained for comparison.
# def _prepare_kmnist(root: str, is_download: bool = True) -> None:
#     """Reuse verified local archives when their extracted IDX files are missing."""
#     raw_folder = os.path.join(root, "KMNIST", "raw")
#     for filename, md5 in datasets.KMNIST.resources:
#         raw_path = os.path.join(raw_folder, filename.removesuffix(".gz"))
#         if os.path.isfile(raw_path):
#             continue
#         archive_path = os.path.join(raw_folder, filename)
#         if not os.path.isfile(archive_path):
#             continue
#         if not check_integrity(archive_path, md5):
#             if is_download:
#                 continue  # torchvision will replace the invalid archive.
#             raise RuntimeError(f"KMNIST archive checksum failed: {archive_path}")
#         extract_archive(archive_path, raw_folder)
#
#
# def _load_kmnist(root: str, train: bool, transform, is_download: bool = True):
#     _prepare_kmnist(root, is_download)
#     attempts = 3 if is_download else 1
#     for attempt in range(attempts):
#         try:
#             return datasets.KMNIST(
#                 root=root,
#                 train=train,
#                 transform=transform,
#                 download=is_download,
#             )
#         except RuntimeError as error:
#             if "Error downloading" not in str(error) or attempt + 1 == attempts:
#                 raise
#             print(f"[KMNIST] Download failed, retrying ({attempt + 2}/{attempts}) ...")
#             time.sleep(attempt + 1)


def _load_kmnist(root: str, train: bool, transform, is_download: bool = True, data=None):
    if data is None:
        data = load_hf_dataset("tanganke/kmnist", root, is_download, name="kmnist")
    return HFImageDataset(
        data["train" if train else "test"], transform=transform, mode="L",
        classes=["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
    )


class DatasetLoader_Kmnist(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        # 默认 transform：ToTensor -> Normalize -> Flatten（与 MNIST 对齐）
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

        self.data_sample_num = 60000
        self.task_type = "cv"

        # Previous loader:
        # self._dataset = datasets.KMNIST(
        #     root=args.root,
        #     train=args.is_train,
        #     transform=args.transform,
        #     download=args.is_download,
        # )
        # self._dataset = _load_kmnist(args.root, args.is_train, args.transform, args.is_download)
        data = load_hf_dataset("tanganke/kmnist", args.root, args.is_download, name="kmnist")
        self._dataset = _load_kmnist(args.root, args.is_train, args.transform, args.is_download, data)

        generator = self.make_generator(args)
        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            generator=generator,
        )

        # 测试集（保持与 MNIST/FMNIST 一致的接口）
        test_transform = getattr(args, "test_transform", None) or args.transform
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size

        # self._test_dataset = datasets.KMNIST(
        #     root=args.root,
        #     train=False,
        #     transform=test_transform,
        #     download=args.is_download,
        # )
        # self._test_dataset = _load_kmnist(args.root, False, test_transform, args.is_download)
        self._test_dataset = _load_kmnist(args.root, False, test_transform, args.is_download, data)
        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return
