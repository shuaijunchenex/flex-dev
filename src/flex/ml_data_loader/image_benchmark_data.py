"""Image sources and split views for the small CNN federated experiments."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.utils import download_url


IMAGENETTE_CLASSES = (
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
)
OFFICE_CLASSES = (
    "back_pack", "bike", "calculator", "headphones", "keyboard",
    "laptop_computer", "monitor", "mouse", "mug", "projector",
)
OFFICE_COUNTS = {"caltech": 1123, "amazon": 958, "webcam": 295, "dslr": 157}
OFFICE_ARCHIVE_URL = (
    "https://codeload.github.com/ChristophRaab/Office_Caltech_DA_Dataset/zip/"
    "91ad5c3f5b7b205083692e2aa62630692596cb1a"
)
OFFICE_ZIP_SHA256 = "8abe3e6f36c38a6c8bb7b3743aaddf633b60915b8e20b6b293da5b704d4ceef6"
DERMA_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
IMAGENETTE_COUNTS = {"train": 9469, "val": 3925}
DERMA_COUNTS = {"train": 7007, "val": 1003, "test": 2005}


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    label: int
    split: str
    domain: str | None
    image: Path | np.ndarray


@dataclass(frozen=True)
class ImageCatalog:
    name: str
    classes: tuple[str, ...]
    records: dict[str, ImageRecord]
    source_sha256: str


def content_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _file_digest(path: Path, name: str = "sha256") -> str:
    digest = hashlib.new(name)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_source(name: str, root: Path) -> Path | None:
    if name == "dermamnist":
        path = root / "dermamnist.npz"
        return path if path.is_file() else None
    if name == "imagenette_v2":
        for path in (root / "imagenette2-160", root):
            if all((path / split).is_dir() for split in IMAGENETTE_COUNTS):
                return path
    elif name == "office_caltech10":
        for path in (root / "office-caltech", root, root / "office_caltech_10"):
            if path.is_dir() and set(OFFICE_COUNTS) <= {
                p.name.lower() for p in path.iterdir() if p.is_dir()
            }:
                return path
    else:
        raise ValueError(f"Unknown image benchmark dataset: {name}")
    return None


def _install_source(name: str, root: Path, *, download: bool) -> None:
    # Extract only into a staging directory. Publish a complete source atomically.
    if not download and not root.is_dir():
        raise FileNotFoundError(root)
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root, prefix=".preparing-") as temp:
        stage = Path(temp)
        if name == "dermamnist":
            filename = "dermamnist.npz"
            if not download:
                raise FileNotFoundError(root / filename)
            download_url(
                "https://zenodo.org/records/10519652/files/dermamnist.npz?download=1",
                str(stage), filename=filename, md5="0744692d530f8e62ec473284d019b0c7",
            )
            (stage / filename).replace(root / filename)
            return

        if name == "imagenette_v2":
            folder = "imagenette2-160"
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
            md5 = "e793b78cc4c9e9a4ccc0c1155377a412"
            archive_name = "imagenette2-160.tgz"
        else:
            folder = "office-caltech"
            url = OFFICE_ARCHIVE_URL
            md5 = None
            archive_name = "office-caltech.zip"
        archive_path = root / archive_name
        if name == "office_caltech10" and (root / "office-caltech.tgz").is_file():
            archive_path = root / "office-caltech.tgz"
        if not archive_path.is_file():
            if not download:
                raise FileNotFoundError(f"Dataset directory or local archive missing at {archive_path}")
            download_url(url, str(stage), filename=archive_name, md5=md5)
            archive_path = stage / archive_name
        elif md5 and _file_digest(archive_path, "md5") != md5:
            raise ValueError(f"Archive checksum mismatch: {archive_path}")
        if archive_path.suffix == ".zip" and _file_digest(archive_path) != OFFICE_ZIP_SHA256:
            raise ValueError(f"Office-Caltech archive differs from the pinned RGB release: {archive_path}")
        extracted = stage / "images"
        extracted.mkdir()
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                for filename in archive.namelist():
                    if not (extracted / filename).resolve().is_relative_to(extracted):
                        raise ValueError(f"Invalid archive path: {filename}")
                archive.extractall(extracted)
        else:
            with tarfile.open(archive_path, mode="r:*") as archive:
                for member in archive:
                    if not member.isdir() and not member.isfile():
                        continue
                    if not (extracted / member.name).resolve().is_relative_to(extracted):
                        raise ValueError(f"Invalid archive path: {member.name}")
                    archive.extract(member, extracted)
        source = _find_source(name, extracted)
        if source is None:
            directories = [p for p in extracted.iterdir() if p.is_dir()]
            if len(directories) == 1:
                source = _find_source(name, directories[0])
        if source is None:
            raise ValueError(f"Unexpected {name} archive layout at {url}")
        try:
            source.rename(root / folder)
        except OSError:
            if _find_source(name, root) is None:
                raise


def prepare_dataset(name: str, root: str | Path, *, download: bool) -> ImageCatalog:
    root = Path(root).resolve()
    source = _find_source(name, root)
    if source is None:
        _install_source(name, root, download=download)
        source = _find_source(name, root)
    if source is None:
        raise FileNotFoundError(f"{name} data not found under {root}. Prepare the dataset first.")
    return _read_catalog(name, source)


def _folder_records(path: Path, classes, *, split: str, domain: str | None = None):
    prefix = domain or split
    aliases = {"backpack": "back_pack", "laptop": "laptop_computer"}
    found = set()
    for directory in sorted(path.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        label_name = directory.name.lower()
        if domain:
            label_name = aliases.get(label_name, label_name)
        if label_name not in classes or label_name in found:
            raise ValueError(f"Unexpected or repeated class directory: {directory}")
        found.add(label_name)
        for image in sorted(directory.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMG_EXTENSIONS and not image.name.startswith("."):
                sample_id = f"{prefix}/{image.relative_to(path).as_posix()}"
                yield ImageRecord(sample_id, classes.index(label_name), split, domain, image)
    if found != set(classes):
        raise ValueError(f"Missing image classes under {path}: {set(classes) - found}")


@lru_cache(maxsize=3)
def _read_catalog(name: str, source: Path) -> ImageCatalog:
    records = []
    if name == "dermamnist":
        classes = DERMA_CLASSES
        if _file_digest(source, "md5") != "0744692d530f8e62ec473284d019b0c7":
            raise ValueError(f"DermaMNIST checksum does not match the official 28px release: {source}")
        with np.load(source, allow_pickle=False) as data:
            for split, count in DERMA_COUNTS.items():
                images = data[f"{split}_images"]
                labels = data[f"{split}_labels"].reshape(-1)
                if images.shape != (count, 28, 28, 3) or images.dtype != np.uint8:
                    raise ValueError(f"Invalid DermaMNIST {split} images: {images.shape}, {images.dtype}")
                if len(labels) != count or set(labels.tolist()) != set(range(len(classes))):
                    raise ValueError(f"Invalid DermaMNIST {split} labels")
                for index, (pixels, label) in enumerate(zip(images, labels)):
                    records.append(ImageRecord(f"{split}/{index}", int(label), split, None, pixels))
        fingerprint = _file_digest(source)
    else:
        if name == "imagenette_v2":
            classes = IMAGENETTE_CLASSES
            for split, expected in IMAGENETTE_COUNTS.items():
                images = list(_folder_records(source / split, classes, split=split))
                if len(images) != expected:
                    raise ValueError(f"{split} has {len(images)} images, expected {expected}")
                records.extend(images)
        else:
            classes = OFFICE_CLASSES
            domain_folders = {p.name.lower(): p for p in source.iterdir() if p.is_dir()}
            for domain, expected in OFFICE_COUNTS.items():
                folder = domain_folders[domain]
                if (folder / "images").is_dir():
                    folder = folder / "images"
                images = list(_folder_records(folder, classes, split="all", domain=domain))
                if len(images) != expected:
                    raise ValueError(f"{domain} has {len(images)} images, expected {expected}")
                records.extend(images)
        fingerprint = content_hash([
            (row.sample_id, row.label, _file_digest(row.image)) for row in records
        ])
    return ImageCatalog(name, classes, {row.sample_id: row for row in records}, fingerprint)


def read_manifest(path: str | Path, catalog: ImageCatalog) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    digest = manifest.pop("manifest_sha256")
    if digest != content_hash(manifest):
        raise ValueError(f"Manifest checksum mismatch: {path}")
    if manifest["dataset"] != catalog.name or manifest["source_sha256"] != catalog.source_sha256:
        raise ValueError(f"Dataset and manifest differ: {path}")
    manifest["manifest_sha256"] = digest
    return manifest


class ImageBenchmarkDataset(Dataset):
    def __init__(self, catalog: ImageCatalog, sample_ids, transform):
        self.catalog = catalog
        self.sample_ids = tuple(sample_ids)
        self.records = [catalog.records[sample_id] for sample_id in self.sample_ids]
        self.targets = [sample.label for sample in self.records]
        self.domains = [sample.domain for sample in self.records] if catalog.name == "office_caltech10" else None
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = self.records[index]
        if isinstance(sample.image, Path):
            with Image.open(sample.image) as source:
                image = source.convert("RGB")
        else:
            image = Image.fromarray(sample.image)
        return self.transform(image), sample.label


def image_transform(name: str, train: bool):
    if name == "dermamnist":
        ops = [transforms.Pad(2, padding_mode="edge")]
    else:
        ops = [transforms.Resize(32, antialias=True), transforms.CenterCrop(32)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )])
    return transforms.Compose(ops)
