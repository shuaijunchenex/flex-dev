from __future__ import annotations

import os
import tarfile
import shutil
import urllib.request
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

'''
Dataset loader for CINIC-10
CINIC-10 is a drop-in CIFAR-10 supplement (32x32 RGB, same 10 classes).
Expected root layout after download/extract:
    <root>/cinic-10/train/   <class_name>/*.png
    <root>/cinic-10/valid/   <class_name>/*.png
    <root>/cinic-10/test/    <class_name>/*.png

Auto-download sources (tried in order):
  1. TensorFlow Datasets (tfds) — pip install tensorflow-datasets tensorflow
  2. HuggingFace Hub            — pip install huggingface_hub
  3. Google Drive               — pip install gdown
  4. Direct HTTPS               — Edinburgh DataShare (~480 MB, no extra deps)

Set is_download: True in yaml to enable auto-download.
'''

# ─── TFDS dataset name ────────────────────────────────────────────────────────
_TFDS_NAME = "cinic10"          # registered in tensorflow-datasets ≥ 4.x
# ─── HuggingFace Hub fallback ─────────────────────────────────────────────────
_HF_REPO_ID  = "bhargavsdesai/cinic10"
_HF_FILENAME = "CINIC-10.tar.gz"
# ─── Google Drive fallback ────────────────────────────────────────────────────
_GDRIVE_ID   = "1G1E1sCcGfkWP1Z7hHyJRr0x3Yd6aJeqZ"
# ─── Direct HTTPS fallback ────────────────────────────────────────────────────
_DIRECT_URL  = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    "?sequence=4&isAllowed=y"
)

# TFDS split name → local folder name mapping
_TFDS_SPLIT_MAP = {"train": "train", "validation": "valid", "test": "test"}

# CINIC-10 class index → class name (same order as CIFAR-10)
_CINIC10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
_CINIC10_STD  = (0.24205776, 0.23828046, 0.25874835)


def _is_cinic10_present(cinic_root: str) -> bool:
    """Return True only when all three splits contain at least one class folder."""
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(cinic_root, split)
        if not os.path.isdir(split_dir):
            return False
        subdirs = [d for d in os.listdir(split_dir)
                   if os.path.isdir(os.path.join(split_dir, d))]
        if len(subdirs) == 0:
            return False
    return True


def _extract_tar(tar_path: str, dest_dir: str, cinic_root: str) -> None:
    """Extract tar.gz and move contents so cinic_root/{train,valid,test} exists."""
    print(f"[CINIC-10] Extracting {tar_path} → {dest_dir} …")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    for candidate in Path(dest_dir).iterdir():
        if candidate.is_dir() and (candidate / "train").exists():
            if str(candidate) != cinic_root:
                if os.path.exists(cinic_root):
                    shutil.rmtree(cinic_root)
                shutil.move(str(candidate), cinic_root)
            break


def _tfds_to_imagefolder(cinic_root: str) -> None:
    """
    Use tensorflow-datasets to download CINIC-10 and write images to
    the ImageFolder layout expected by torchvision:
        cinic_root/{train,valid,test}/<class_name>/<idx>.png
    """
    import tensorflow_datasets as tfds  # type: ignore
    from PIL import Image  # type: ignore
    import numpy as np

    print("[CINIC-10] Downloading via TensorFlow Datasets (tfds) …")

    # tfds knows cinic10 with splits: train / validation / test
    ds_all, info = tfds.load(
        _TFDS_NAME,
        split=list(_TFDS_SPLIT_MAP.keys()),
        as_supervised=True,    # yields (image, label) tuples
        with_info=True,
        shuffle_files=False,
    )

    counters: dict[str, dict[int, int]] = {}

    for tfds_split, local_split in _TFDS_SPLIT_MAP.items():
        ds = ds_all[list(_TFDS_SPLIT_MAP.keys()).index(tfds_split)]
        counters[local_split] = {i: 0 for i in range(len(_CINIC10_CLASSES))}

        # Pre-create class dirs
        for cls in _CINIC10_CLASSES:
            os.makedirs(os.path.join(cinic_root, local_split, cls), exist_ok=True)

        print(f"[CINIC-10]   Writing split '{local_split}' …")
        for image_tensor, label_tensor in tfds.as_numpy(ds):
            label = int(label_tensor)
            cls   = _CINIC10_CLASSES[label]
            idx   = counters[local_split][label]
            counters[local_split][label] += 1
            img_path = os.path.join(cinic_root, local_split, cls, f"{idx:06d}.png")
            Image.fromarray(image_tensor.astype(np.uint8)).save(img_path)

        total = sum(counters[local_split].values())
        print(f"[CINIC-10]   '{local_split}' done — {total} images.")


def _download_cinic10(cinic_root: str) -> None:
    """Try TFDS → HuggingFace Hub → Google Drive → direct URL to obtain CINIC-10."""
    dest_dir = str(Path(cinic_root).parent)
    tar_path = os.path.join(dest_dir, "CINIC-10.tar.gz")

    # ── Source 1: TensorFlow Datasets ────────────────────────────────────────
    try:
        _tfds_to_imagefolder(cinic_root)
        if _is_cinic10_present(cinic_root):
            print("[CINIC-10] ✅ Downloaded & written via TFDS.")
            return
    except Exception as e:
        print(f"[CINIC-10] TFDS failed ({e}), trying HuggingFace Hub …")

    # ── Source 2: HuggingFace Hub ─────────────────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        # Support HF mirror for users behind restrictive networks (e.g. China mainland).
        # Set env var HF_ENDPOINT=https://hf-mirror.com before running, or
        # the code will auto-try the mirror if the official endpoint is unreachable.
        _HF_ENDPOINTS = [
            os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            "https://hf-mirror.com",
        ]
        downloaded = None
        last_err = None
        for endpoint in _HF_ENDPOINTS:
            if endpoint is None:
                continue
            try:
                print(f"[CINIC-10] Downloading via HuggingFace Hub ({endpoint}) …")
                downloaded = hf_hub_download(
                    repo_id=_HF_REPO_ID,
                    filename=_HF_FILENAME,
                    repo_type="dataset",
                    local_dir=dest_dir,
                    endpoint=endpoint,
                )
                break  # success – exit the endpoint loop
            except Exception as _e:
                last_err = _e
                print(f"[CINIC-10]   {endpoint} failed ({_e}), trying next …")
        if downloaded is None:
            raise last_err or RuntimeError("All HF endpoints failed.")
        shutil.copy(downloaded, tar_path)
        _extract_tar(tar_path, dest_dir, cinic_root)
        if _is_cinic10_present(cinic_root):
            print("[CINIC-10] ✅ Downloaded & extracted via HuggingFace Hub.")
            return
    except Exception as e:
        print(f"[CINIC-10] HuggingFace Hub failed ({e}), trying Google Drive …")

    # ── Source 3: Google Drive via gdown ──────────────────────────────────────
    try:
        import gdown  # type: ignore
        print("[CINIC-10] Downloading via Google Drive (gdown) …")
        gdown.download(id=_GDRIVE_ID, output=tar_path, quiet=False, fuzzy=True)
        _extract_tar(tar_path, dest_dir, cinic_root)
        if _is_cinic10_present(cinic_root):
            print("[CINIC-10] ✅ Downloaded & extracted via Google Drive.")
            return
    except Exception as e:
        print(f"[CINIC-10] Google Drive failed ({e}), trying direct URL …")

    # ── Source 4: Direct HTTPS ────────────────────────────────────────────────
    try:
        print("[CINIC-10] Downloading via direct URL (~480 MB) …")

        def _reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(count * block_size * 100 / total_size))
                print(f"\r[CINIC-10]  {pct}%", end="", flush=True)

        urllib.request.urlretrieve(_DIRECT_URL, tar_path, reporthook=_reporthook)
        print()  # newline after progress
        _extract_tar(tar_path, dest_dir, cinic_root)
        if _is_cinic10_present(cinic_root):
            print("[CINIC-10] ✅ Downloaded & extracted via direct URL.")
            return
    except Exception as e:
        print(f"[CINIC-10] Direct URL failed ({e}).")

    # ── All four sources exhausted — offer tfds install as last resort ────────
    tfds_available = False
    try:
        import tensorflow_datasets  # type: ignore # noqa: F401
        tfds_available = True
    except ImportError:
        pass

    if not tfds_available:
        print("[CINIC-10] All four download sources failed.")
        answer = input(
            "[CINIC-10] tensorflow-datasets (tfds) is not installed. "
            "Would you like to install it and try downloading via tfds? [y/N]: "
        )
        if answer.strip().lower() in ("y", "yes"):
            print("[CINIC-10] Installing tensorflow-datasets …")
            import subprocess
            import sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "tensorflow-datasets", "tensorflow"]
            )
            try:
                _tfds_to_imagefolder(cinic_root)
                if _is_cinic10_present(cinic_root):
                    print("[CINIC-10] ✅ Downloaded & written via TFDS.")
                    return
            except Exception as e:
                print(f"[CINIC-10] TFDS download also failed after install ({e}).")

    raise RuntimeError(
        "[CINIC-10] All download sources failed. "
        "Please download manually from https://datashare.ed.ac.uk/handle/10283/3192 "
        f"and extract to: {cinic_root}"
    )


class DatasetLoader_Cinic10(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        cinic_root = os.path.join(args.root, "cinic-10")

        # Auto-download when is_download=True and dataset not yet present
        if getattr(args, "is_download", False) and not _is_cinic10_present(cinic_root):
            os.makedirs(cinic_root, exist_ok=True)
            _download_cinic10(cinic_root)

        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(_CINIC10_MEAN, _CINIC10_STD),
            ])

        test_transform = getattr(args, "test_transform", None)
        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_CINIC10_MEAN, _CINIC10_STD),
            ])
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size

        # Use "train" split for training; valid and test are treated as evaluation sets.
        # args.split controls which folder is used:
        #   ""  / "train" → train folder  (default)
        #   "valid"       → valid folder
        #   "test"        → test folder
        train_split = args.split if args.split in ("train", "valid", "test") else "train"
        test_split  = "test"

        # args.is_train=True  → load train split (default behaviour)
        # args.is_train=False → skip train, only load test split
        if getattr(args, "is_train", True):
            train_dir = os.path.join(cinic_root, train_split)
            self._dataset = datasets.ImageFolder(root=train_dir, transform=args.transform)
            self.data_sample_num = len(self._dataset)  # 90,000 for default train split
            self.task_type = "cv"
            generator = self.make_generator(args)
            self._data_loader = DataLoader(
                self._dataset,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                num_workers=args.num_workers,
                generator=generator,
            )

        # Always load test split so evaluation is available regardless of is_train
        if True:
            test_dir = os.path.join(cinic_root, test_split)
            self._test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
            self._test_data_loader = DataLoader(
                self._test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
        return
