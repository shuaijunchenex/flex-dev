from __future__ import annotations

# import os
# import tarfile
# import shutil
# import stat
# import tempfile
# import uuid
# import urllib.request
# from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# import torchvision.datasets as datasets

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from pathlib import PurePosixPath

from ._hf_dataset import load_hf_dataset
from ._hf_image_dataset import HFImageDataset

# Previous download sources retained for comparison; active loading uses HF only.
# '''
# Dataset loader for CINIC-10
# CINIC-10 is a drop-in CIFAR-10 supplement (32x32 RGB, same 10 classes).
# Expected root layout after download/extract:
#     <root>/cinic-10/train/   <class_name>/*.png
#     <root>/cinic-10/valid/   <class_name>/*.png
#     <root>/cinic-10/test/    <class_name>/*.png
#
# Auto-download sources (tried in order):
#   1. TensorFlow Datasets (tfds) — pip install tensorflow-datasets tensorflow
#   2. HuggingFace Hub            — pip install huggingface_hub
#   3. Google Drive               — pip install gdown
#   4. Direct HTTPS               — Edinburgh DataShare (~480 MB, no extra deps)
#
# Set is_download: True in yaml to enable auto-download.
# '''
#
# # ─── TFDS dataset name ────────────────────────────────────────────────────────
# _TFDS_NAME = "cinic10"          # registered in tensorflow-datasets ≥ 4.x
# # ─── HuggingFace Hub fallback ─────────────────────────────────────────────────
# _HF_REPO_ID  = "bhargavsdesai/cinic10"
# _HF_FILENAME = "CINIC-10.tar.gz"
# # ─── Google Drive fallback ────────────────────────────────────────────────────
# _GDRIVE_ID   = "1G1E1sCcGfkWP1Z7hHyJRr0x3Yd6aJeqZ"
# # ─── Direct HTTPS fallback ────────────────────────────────────────────────────
# _DIRECT_URL  = (
#     "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
#     "?sequence=4&isAllowed=y"
# )
#
# # TFDS split name → local folder name mapping
# _TFDS_SPLIT_MAP = {"train": "train", "validation": "valid", "test": "test"}
#
# # CINIC-10 class index → class name (same order as CIFAR-10)
_CINIC10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
_CINIC10_STD  = (0.24205776, 0.23828046, 0.25874835)


# Previous local extraction/download helpers retained; not used by HF loading.
# # Previous implementation retained for comparison.
# # def _is_cinic10_present(cinic_root: str) -> bool:
# #     """Return True only when all three splits contain at least one class folder."""
# #     for split in ("train", "valid", "test"):
# #         split_dir = os.path.join(cinic_root, split)
# #         if not os.path.isdir(split_dir):
# #             return False
# #         subdirs = [d for d in os.listdir(split_dir)
# #                    if os.path.isdir(os.path.join(split_dir, d))]
# #         if len(subdirs) == 0:
# #             return False
# #     return True
# #
# #
# def _is_cinic10_present(cinic_root: str) -> bool:
#     """Check all splits/classes, preserving permission errors for the caller."""
#     def raise_walk_error(error):
#         raise error
#
#     for split in ("train", "valid", "test"):
#         split_dir = os.path.join(cinic_root, split)
#         try:
#             with os.scandir(split_dir) as entries:
#                 class_dirs = {entry.name for entry in entries if entry.is_dir()}
#             if class_dirs != set(_CINIC10_CLASSES):
#                 return False
#             for cls in _CINIC10_CLASSES:
#                 for _, _, names in os.walk(
#                     os.path.join(split_dir, cls), onerror=raise_walk_error
#                 ):
#                     if any(name.lower().endswith(datasets.folder.IMG_EXTENSIONS) for name in names):
#                         break
#                 else:
#                     return False
#         except (FileNotFoundError, NotADirectoryError):
#             return False
#     return True
#
#
# def _find_cinic10_root(folder: Path) -> Path:
#     """Accept flat archives or an enclosing directory inside fresh staging."""
#     if _is_cinic10_present(str(folder)):
#         return folder
#     for candidate in sorted(folder.iterdir()):
#         if _is_cinic10_present(str(candidate)):
#             return candidate
#     raise ValueError("[CINIC-10] Archive does not contain all splits and nonempty classes.")
#
#
# def _publish_cinic10(staged_root: Path, cinic_root: str) -> None:
#     """Publish validated staging, retaining an existing incomplete cache."""
#     target = Path(cinic_root).absolute()
#     parent = target.parent.resolve()
#     target = parent / target.name
#     staged_root = staged_root.resolve()
#     # Every directory move stays within the explicitly requested dataset parent.
#     if parent not in staged_root.parents or staged_root == target:
#         raise ValueError("[CINIC-10] Staging must be separate and inside the dataset parent.")
#     if not _is_cinic10_present(str(staged_root)):
#         raise ValueError("[CINIC-10] Refusing to publish incomplete data.")
#     if _is_cinic10_present(str(target)):
#         return
#     backup = None
#     try:
#         target_info = target.lstat()
#     except FileNotFoundError:
#         pass
#     else:
#         if stat.S_ISLNK(target_info.st_mode):
#             raise ValueError("[CINIC-10] Refusing to replace a symlink dataset root.")
#         backup = parent / f"{target.name}.previous-{uuid.uuid4().hex}"
#         target.rename(backup)
#     try:
#         staged_root.rename(target)
#     except Exception:
#         if backup is not None:
#             backup.rename(target)
#         raise
#     if backup is not None:
#         print(f"[CINIC-10] Previous incomplete cache preserved at: {backup}")
#
# # Previous implementation retained for comparison.
# # def _extract_tar(tar_path: str, dest_dir: str, cinic_root: str) -> None:
# #     """Extract tar.gz and move contents so cinic_root/{train,valid,test} exists."""
# #     print(f"[CINIC-10] Extracting {tar_path} → {dest_dir} …")
# #     with tarfile.open(tar_path, "r:gz") as tf:
# #         tf.extractall(dest_dir)
# #     for candidate in Path(dest_dir).iterdir():
# #         if candidate.is_dir() and (candidate / "train").exists():
# #             if str(candidate) != cinic_root:
# #                 if os.path.exists(cinic_root):
# #                     shutil.rmtree(cinic_root)
# #                 shutil.move(str(candidate), cinic_root)
# #             break
# #
# #
# def _extract_tar(tar_path: str, dest_dir: str, cinic_root: str) -> None:
#     """Extract in fresh staging; normalize flat/wrapped archives after validation."""
#     parent = Path(dest_dir).resolve()
#     target = Path(cinic_root).absolute()
#     if target.parent.resolve() != parent:
#         raise ValueError("[CINIC-10] Dataset root must be directly inside the extraction parent.")
#     parent.mkdir(parents=True, exist_ok=True)
#     print(f"[CINIC-10] Extracting {tar_path} into fresh staging ...")
#     with tempfile.TemporaryDirectory(prefix=".cinic10-extract-", dir=parent) as temp_dir:
#         staging = Path(temp_dir).resolve()
#         if staging.parent != parent:
#             raise ValueError("[CINIC-10] Invalid extraction staging directory.")
#         with tarfile.open(tar_path, "r:gz") as tf:
#             for member in tf.getmembers():
#                 destination = (staging / member.name).resolve()
#                 if destination != staging and staging not in destination.parents:
#                     raise ValueError(f"[CINIC-10] Archive path escapes staging: {member.name}")
#                 if not member.isfile() and not member.isdir():
#                     raise ValueError(f"[CINIC-10] Unsupported archive entry: {member.name}")
#             if hasattr(tarfile, "data_filter"):
#                 tf.extractall(staging, filter="data")
#             else:
#                 # Earlier Python 3.10 patches rely on the explicit checks above.
#                 tf.extractall(staging)
#         _publish_cinic10(_find_cinic10_root(staging), str(target))
#
# def _tfds_to_imagefolder(cinic_root: str) -> None:
#     """
#     Use tensorflow-datasets to download CINIC-10 and write images to
#     the ImageFolder layout expected by torchvision:
#         cinic_root/{train,valid,test}/<class_name>/<idx>.png
#     """
#     import tensorflow_datasets as tfds  # type: ignore
#     from PIL import Image  # type: ignore
#     import numpy as np
#
#     print("[CINIC-10] Downloading via TensorFlow Datasets (tfds) …")
#
#     # tfds knows cinic10 with splits: train / validation / test
#     ds_all, info = tfds.load(
#         _TFDS_NAME,
#         split=list(_TFDS_SPLIT_MAP.keys()),
#         as_supervised=True,    # yields (image, label) tuples
#         with_info=True,
#         shuffle_files=False,
#     )
#
#     counters: dict[str, dict[int, int]] = {}
#
#     for tfds_split, local_split in _TFDS_SPLIT_MAP.items():
#         ds = ds_all[list(_TFDS_SPLIT_MAP.keys()).index(tfds_split)]
#         counters[local_split] = {i: 0 for i in range(len(_CINIC10_CLASSES))}
#
#         # Pre-create class dirs
#         for cls in _CINIC10_CLASSES:
#             os.makedirs(os.path.join(cinic_root, local_split, cls), exist_ok=True)
#
#         print(f"[CINIC-10]   Writing split '{local_split}' …")
#         for image_tensor, label_tensor in tfds.as_numpy(ds):
#             label = int(label_tensor)
#             cls   = _CINIC10_CLASSES[label]
#             idx   = counters[local_split][label]
#             counters[local_split][label] += 1
#             img_path = os.path.join(cinic_root, local_split, cls, f"{idx:06d}.png")
#             Image.fromarray(image_tensor.astype(np.uint8)).save(img_path)
#
#         total = sum(counters[local_split].values())
#         print(f"[CINIC-10]   '{local_split}' done — {total} images.")
#
#
# # Previous implementation retained for comparison.
# # def _download_cinic10(cinic_root: str) -> None:
# #     """Try TFDS → HuggingFace Hub → Google Drive → direct URL to obtain CINIC-10."""
# #     dest_dir = str(Path(cinic_root).parent)
# #     tar_path = os.path.join(dest_dir, "CINIC-10.tar.gz")
# #
# #     # ── Source 1: TensorFlow Datasets ────────────────────────────────────────
# #     try:
# #         _tfds_to_imagefolder(cinic_root)
# #         if _is_cinic10_present(cinic_root):
# #             print("[CINIC-10] ✅ Downloaded & written via TFDS.")
# #             return
# #     except Exception as e:
# #         print(f"[CINIC-10] TFDS failed ({e}), trying HuggingFace Hub …")
# #
# #     # ── Source 2: HuggingFace Hub ─────────────────────────────────────────────
# #     try:
# #         from huggingface_hub import hf_hub_download  # type: ignore
# #         # Support HF mirror for users behind restrictive networks (e.g. China mainland).
# #         # Set env var HF_ENDPOINT=https://hf-mirror.com before running, or
# #         # the code will auto-try the mirror if the official endpoint is unreachable.
# #         _HF_ENDPOINTS = [
# #             os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
# #             "https://hf-mirror.com",
# #         ]
# #         downloaded = None
# #         last_err = None
# #         for endpoint in _HF_ENDPOINTS:
# #             if endpoint is None:
# #                 continue
# #             try:
# #                 print(f"[CINIC-10] Downloading via HuggingFace Hub ({endpoint}) …")
# #                 downloaded = hf_hub_download(
# #                     repo_id=_HF_REPO_ID,
# #                     filename=_HF_FILENAME,
# #                     repo_type="dataset",
# #                     local_dir=dest_dir,
# #                     endpoint=endpoint,
# #                 )
# #                 break  # success – exit the endpoint loop
# #             except Exception as _e:
# #                 last_err = _e
# #                 print(f"[CINIC-10]   {endpoint} failed ({_e}), trying next …")
# #         if downloaded is None:
# #             raise last_err or RuntimeError("All HF endpoints failed.")
# #         shutil.copy(downloaded, tar_path)
# #         _extract_tar(tar_path, dest_dir, cinic_root)
# #         if _is_cinic10_present(cinic_root):
# #             print("[CINIC-10] ✅ Downloaded & extracted via HuggingFace Hub.")
# #             return
# #     except Exception as e:
# #         print(f"[CINIC-10] HuggingFace Hub failed ({e}), trying Google Drive …")
# #
# #     # ── Source 3: Google Drive via gdown ──────────────────────────────────────
# #     try:
# #         import gdown  # type: ignore
# #         print("[CINIC-10] Downloading via Google Drive (gdown) …")
# #         gdown.download(id=_GDRIVE_ID, output=tar_path, quiet=False, fuzzy=True)
# #         _extract_tar(tar_path, dest_dir, cinic_root)
# #         if _is_cinic10_present(cinic_root):
# #             print("[CINIC-10] ✅ Downloaded & extracted via Google Drive.")
# #             return
# #     except Exception as e:
# #         print(f"[CINIC-10] Google Drive failed ({e}), trying direct URL …")
# #
# #     # ── Source 4: Direct HTTPS ────────────────────────────────────────────────
# #     try:
# #         print("[CINIC-10] Downloading via direct URL (~480 MB) …")
# #
# #         def _reporthook(count, block_size, total_size):
# #             if total_size > 0:
# #                 pct = min(100, int(count * block_size * 100 / total_size))
# #                 print(f"\r[CINIC-10]  {pct}%", end="", flush=True)
# #
# #         urllib.request.urlretrieve(_DIRECT_URL, tar_path, reporthook=_reporthook)
# #         print()  # newline after progress
# #         _extract_tar(tar_path, dest_dir, cinic_root)
# #         if _is_cinic10_present(cinic_root):
# #             print("[CINIC-10] ✅ Downloaded & extracted via direct URL.")
# #             return
# #     except Exception as e:
# #         print(f"[CINIC-10] Direct URL failed ({e}).")
# #
# #     # ── All four sources exhausted — offer tfds install as last resort ────────
# #     tfds_available = False
# #     try:
# #         import tensorflow_datasets  # type: ignore # noqa: F401
# #         tfds_available = True
# #     except ImportError:
# #         pass
# #
# #     if not tfds_available:
# #         print("[CINIC-10] All four download sources failed.")
# #         answer = input(
# #             "[CINIC-10] tensorflow-datasets (tfds) is not installed. "
# #             "Would you like to install it and try downloading via tfds? [y/N]: "
# #         )
# #         if answer.strip().lower() in ("y", "yes"):
# #             print("[CINIC-10] Installing tensorflow-datasets …")
# #             import subprocess
# #             import sys
# #             subprocess.check_call(
# #                 [sys.executable, "-m", "pip", "install", "tensorflow-datasets", "tensorflow"]
# #             )
# #             try:
# #                 _tfds_to_imagefolder(cinic_root)
# #                 if _is_cinic10_present(cinic_root):
# #                     print("[CINIC-10] ✅ Downloaded & written via TFDS.")
# #                     return
# #             except Exception as e:
# #                 print(f"[CINIC-10] TFDS download also failed after install ({e}).")
# #
# #     raise RuntimeError(
# #         "[CINIC-10] All download sources failed. "
# #         "Please download manually from https://datashare.ed.ac.uk/handle/10283/3192 "
# #         f"and extract to: {cinic_root}"
# #     )
# #
# #
# def _download_cinic10(cinic_root: str) -> None:
#     """Prefer local data/archive, then try the existing four download sources."""
#     target = Path(cinic_root).absolute()
#     parent = target.parent.resolve()
#     target = parent / target.name
#     if _is_cinic10_present(str(target)):
#         return
#     parent.mkdir(parents=True, exist_ok=True)
#     tar_path = parent / "CINIC-10.tar.gz"
#
#     # Recover flat extraction and common wrapper names without moving source data.
#     for cached_root in (parent, parent / "CINIC-10", parent / "cinic10", parent / "CINIC10"):
#         if cached_root == target or not _is_cinic10_present(str(cached_root)):
#             continue
#         with tempfile.TemporaryDirectory(prefix=".cinic10-cache-", dir=parent) as temp_dir:
#             staging = Path(temp_dir).resolve()
#             if staging.parent != parent:
#                 raise ValueError("[CINIC-10] Invalid cache staging directory.")
#             for split in ("train", "valid", "test"):
#                 shutil.copytree(cached_root / split, staging / split)
#             _publish_cinic10(staging, str(target))
#         print(f"[CINIC-10] Reused extracted cache from: {cached_root}")
#         return
#
#     try:
#         tar_path.stat()
#     except FileNotFoundError:
#         pass
#     else:
#         try:
#             _extract_tar(str(tar_path), str(parent), str(target))
#             print("[CINIC-10] Reused cached archive.")
#             return
#         except PermissionError:
#             raise
#         except (tarfile.TarError, ValueError, EOFError) as e:
#             print(f"[CINIC-10] Cached archive is unusable ({e}); preserving it and trying sources.")
#
#     def install_archive(downloaded):
#         _extract_tar(str(downloaded), str(parent), str(target))
#         # Keep an existing archive; do not overwrite cache while trying fallbacks.
#         try:
#             tar_path.stat()
#         except FileNotFoundError:
#             if Path(downloaded).resolve() != tar_path.resolve():
#                 shutil.copy2(downloaded, tar_path)
#
#     # Source 1: TensorFlow Datasets. Keep partial writes out of the real cache.
#     try:
#         with tempfile.TemporaryDirectory(prefix=".cinic10-tfds-", dir=parent) as temp_dir:
#             staging = Path(temp_dir).resolve()
#             if staging.parent != parent:
#                 raise ValueError("[CINIC-10] Invalid TFDS staging directory.")
#             _tfds_to_imagefolder(str(staging))
#             _publish_cinic10(staging, str(target))
#         print("[CINIC-10] Downloaded & written via TFDS.")
#         return
#     except PermissionError:
#         raise
#     except Exception as e:
#         print(f"[CINIC-10] TFDS failed ({e}), trying HuggingFace Hub ...")
#
#     # Source 2: HuggingFace Hub, including the existing mirror fallback.
#     try:
#         from huggingface_hub import hf_hub_download  # type: ignore
#         endpoints = [
#             os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
#             "https://hf-mirror.com",
#         ]
#         with tempfile.TemporaryDirectory(prefix=".cinic10-hf-", dir=parent) as temp_dir:
#             local_tar = Path(temp_dir) / "CINIC-10.tar.gz"
#             downloaded = None
#             last_err = None
#             for endpoint in endpoints:
#                 try:
#                     print(f"[CINIC-10] Downloading via HuggingFace Hub ({endpoint}) ...")
#                     downloaded = hf_hub_download(
#                         repo_id=_HF_REPO_ID, filename=_HF_FILENAME,
#                         repo_type="dataset", local_dir=temp_dir, endpoint=endpoint,
#                     )
#                     break
#                 except PermissionError:
#                     raise
#                 except Exception as e:
#                     last_err = e
#                     print(f"[CINIC-10] {endpoint} failed ({e}), trying next ...")
#             if downloaded is None:
#                 raise last_err or RuntimeError("All HF endpoints failed.")
#             # HF local_dir normally already returns this very archive path.
#             if Path(downloaded).resolve() != local_tar.resolve():
#                 shutil.copy2(downloaded, local_tar)
#             install_archive(local_tar)
#         print("[CINIC-10] Downloaded & extracted via HuggingFace Hub.")
#         return
#     except PermissionError:
#         raise
#     except Exception as e:
#         print(f"[CINIC-10] HuggingFace Hub failed ({e}), trying Google Drive ...")
#
#     # Source 3: Google Drive via gdown.
#     try:
#         import gdown  # type: ignore
#         with tempfile.TemporaryDirectory(prefix=".cinic10-gdrive-", dir=parent) as temp_dir:
#             local_tar = Path(temp_dir) / "CINIC-10.tar.gz"
#             print("[CINIC-10] Downloading via Google Drive (gdown) ...")
#             gdown.download(id=_GDRIVE_ID, output=str(local_tar), quiet=False, fuzzy=True)
#             install_archive(local_tar)
#         print("[CINIC-10] Downloaded & extracted via Google Drive.")
#         return
#     except PermissionError:
#         raise
#     except Exception as e:
#         print(f"[CINIC-10] Google Drive failed ({e}), trying direct URL ...")
#
#     # Source 4: Direct HTTPS.
#     try:
#         def reporthook(count, block_size, total_size):
#             if total_size > 0:
#                 pct = min(100, int(count * block_size * 100 / total_size))
#                 print(f"\r[CINIC-10]  {pct}%", end="", flush=True)
#
#         with tempfile.TemporaryDirectory(prefix=".cinic10-http-", dir=parent) as temp_dir:
#             local_tar = Path(temp_dir) / "CINIC-10.tar.gz"
#             print("[CINIC-10] Downloading via direct URL ...")
#             urllib.request.urlretrieve(_DIRECT_URL, str(local_tar), reporthook=reporthook)
#             print()
#             install_archive(local_tar)
#         print("[CINIC-10] Downloaded & extracted via direct URL.")
#         return
#     except PermissionError:
#         raise
#     except Exception as e:
#         print(f"[CINIC-10] Direct URL failed ({e}).")
#
#     # Batch jobs must finish with an actionable error, never input() or pip install.
#     raise RuntimeError(
#         "[CINIC-10] All download sources failed. "
#         "Please download manually from https://datashare.ed.ac.uk/handle/10283/3192 "
#         f"and extract to: {target}"
#     )
#
def _cinic10_image_dataset(split, transform, split_name):
    """Retain HF order unless complete original-layout path metadata can sort it."""
    from datasets import Image as HFImage

    names = getattr(split.features.get("label"), "names", None)
    if names is not None and list(names) != _CINIC10_CLASSES:
        raise ValueError("[CINIC-10] HF class labels differ from the expected ImageFolder labels.")

    order = None
    if isinstance(split.features.get("image"), HFImage):
        raw = split.cast_column("image", HFImage(decode=False))
        # Flatten only the Arrow metadata: do not decode or read image bytes here.
        paths = raw.flatten()["image.path"]
        labels = raw["label"]
        keys = []
        aliases = {"validation", "valid"} if split_name == "validation" else {split_name}
        for path, label in zip(paths, labels):
            if not isinstance(path, str) or not path:
                break
            label = int(label)
            if label < 0 or label >= len(_CINIC10_CLASSES):
                raise ValueError("[CINIC-10] HF label is outside the expected classes.")
            parts = PurePosixPath(path.replace("\\", "/")).parts
            class_name = _CINIC10_CLASSES[label]
            relative = None
            # Accept explicit class/file paths or a recognized split/class/file.
            if len(parts) >= 2 and parts[0] == class_name:
                relative = parts[1:]
            else:
                for i in range(len(parts) - 2):
                    if parts[i] in aliases and parts[i + 1] == class_name:
                        relative = parts[i + 2:]
                        break
            if (not relative or not relative[-1].lower().endswith(".png")
                    or any(part in (".", "..") for part in relative)):
                break
            # ImageFolder sorts walked directories, then filenames inside each.
            keys.append((label, "/".join(relative[:-1]), relative[-1]))
        if keys and len(keys) == len(split) and len(set(keys)) == len(keys):
            order = sorted(range(len(keys)), key=keys.__getitem__)

    if order is not None:
        split = split.select(order)
    result = HFImageDataset(split, transform=transform, mode="RGB", classes=_CINIC10_CLASSES)
    if order is not None:
        result.source_order = "imagefolder_paths"
    print(
        f"[CINIC-10] {split_name}: source_order={result.source_order}; "
        "historical sample/index equivalence remains unverified."
    )
    return result


class DatasetLoader_Cinic10(DatasetLoader):
    def __init__(self):
        super().__init__()

    # Previous ImageFolder implementation retained for comparison.
    # # override
    # def _create_inner(self, args: DatasetLoaderArgs) -> None:
    #     cinic_root = os.path.join(args.root, "cinic-10")
    #
    #     # Auto-download when is_download=True and dataset not yet present
    #     # if getattr(args, "is_download", False) and not _is_cinic10_present(cinic_root):
    #     #     os.makedirs(cinic_root, exist_ok=True)
    #     #     _download_cinic10(cinic_root)
    #     if getattr(args, "is_download", False) and not _is_cinic10_present(cinic_root):
    #         _download_cinic10(cinic_root)
    #
    #     if args.transform is None:
    #         args.transform = transforms.Compose([
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.ToTensor(),
    #             transforms.Normalize(_CINIC10_MEAN, _CINIC10_STD),
    #         ])
    #
    #     test_transform = getattr(args, "test_transform", None)
    #     if test_transform is None:
    #         test_transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize(_CINIC10_MEAN, _CINIC10_STD),
    #         ])
    #     test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size
    #
    #     # Use "train" split for training; valid and test are treated as evaluation sets.
    #     # args.split controls which folder is used:
    #     #   ""  / "train" → train folder  (default)
    #     #   "valid"       → valid folder
    #     #   "test"        → test folder
    #     train_split = args.split if args.split in ("train", "valid", "test") else "train"
    #     test_split  = "test"
    #
    #     # args.is_train=True  → load train split (default behaviour)
    #     # args.is_train=False → skip train, only load test split
    #     if getattr(args, "is_train", True):
    #         train_dir = os.path.join(cinic_root, train_split)
    #         self._dataset = datasets.ImageFolder(root=train_dir, transform=args.transform)
    #         self.data_sample_num = len(self._dataset)  # 90,000 for default train split
    #         self.task_type = "cv"
    #         generator = self.make_generator(args)
    #         self._data_loader = DataLoader(
    #             self._dataset,
    #             batch_size=args.batch_size,
    #             shuffle=args.shuffle,
    #             num_workers=args.num_workers,
    #             generator=generator,
    #         )
    #
    #     # Always load test split so evaluation is available regardless of is_train
    #     if True:
    #         test_dir = os.path.join(cinic_root, test_split)
    #         self._test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    #         self._test_data_loader = DataLoader(
    #             self._test_dataset,
    #             batch_size=test_batch_size,
    #             shuffle=False,
    #             num_workers=args.num_workers,
    #         )
    #     return

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
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

        hf_data = load_hf_dataset(
            "flwrlabs/cinic10", args.root, is_download=getattr(args, "is_download", True)
        )
        hf_splits = {"train": "train", "valid": "validation", "test": "test"}

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
            hf_train_split = hf_splits[train_split]
            self._dataset = _cinic10_image_dataset(
                hf_data[hf_train_split], args.transform, hf_train_split
            )
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
            hf_test_split = hf_splits[test_split]
            self._test_dataset = _cinic10_image_dataset(
                hf_data[hf_test_split], test_transform, hf_test_split
            )
            self._test_data_loader = DataLoader(
                self._test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
        return
