from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset


class HFImageDataset(Dataset):
    """Map HF image rows to torchvision-style image/label pairs lazily."""

    def __init__(self, split, transform=None, mode="RGB", classes=None):
        from datasets import Image as HFImage

        self.transform = transform
        self.target_transform = None
        self.mode = mode
        self.source_order = "hf"
        self.historical_equivalence_verified = False
        features = getattr(split, "features", {})
        if isinstance(features.get("image"), HFImage):
            split = split.cast_column("image", HFImage(decode=False))
        self._split = split
        # Reading labels never decodes images or consumes augmentation RNG.
        self.targets = [int(label) for label in split["label"]]
        names = getattr(features.get("label"), "names", None)
        if classes is None:
            classes = names or [str(i) for i in range(max(self.targets, default=-1) + 1)]
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        if any(label < 0 or label >= len(self.classes) for label in self.targets):
            raise ValueError("HF image labels are outside the declared classes.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self._split[int(index)]["image"]
        if isinstance(image, Mapping):
            encoded = image.get("bytes")
            source = BytesIO(encoded) if encoded is not None else image.get("path")
            if source is None:
                raise ValueError("HF image row has neither embedded bytes nor a path.")
            with Image.open(source) as decoded:
                image = decoded.convert(self.mode)
        elif isinstance(image, Image.Image):
            image = image.convert(self.mode)
        else:
            raise TypeError("HF image rows must contain PIL images or encoded image mappings.")
        label = self.targets[int(index)]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
