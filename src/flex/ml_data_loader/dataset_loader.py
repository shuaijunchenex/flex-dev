from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from .dataset_loader_args import DatasetLoaderArgs
import operator


class DatasetLoader(ABC):
    """
    " Data set loader abstract class
    """

    def __init__(self):
        self._dataset_type: str = ""                # Dataset type
        self._data_loader: Optional[DataLoader] = None    # Training data loader
        self._test_data_loader: Optional[DataLoader] = None
        self._dataset: Dataset | None = None
        self._args: DatasetLoaderArgs | None = None
        self._after_create_fn: Callable[[DatasetLoader], None] | None = None
        return

    # --------------------------------------------------
    @property
    def dataset_type(self): return self._dataset_type

    @property
    def data_loader(self) -> DataLoader:
        if self._data_loader is not None:
            return self._data_loader 
        else:
            raise ValueError("ERROR: DatasetLoader's data_loader is None.")

    @property
    def test_data_loader(self) -> DataLoader:
        if self._test_data_loader is not None:
            return self._test_data_loader
        else:
            raise ValueError("ERROR: DatasetLoader's test_data_loader is None.")

    @property
    def data_set(self): return self._dataset

    @property
    def is_loaded(self): return self._data_loader is not None

    @property
    def args(self): return self._args

    # --------------------------------------------------
    def create(self, args: DatasetLoaderArgs, fn: Callable[[DatasetLoader], None]|None = None) -> DatasetLoader:
        """
        Create Dataset Loader
        """
        self._args = args
        self._create_inner(args)  # create dataset loader
        if fn is not None:
            self._after_create_fn = fn
            fn(self)

        #self._get_data_length()
        # if self._data_loader.data_sample_num is not None:
        #     self.data_sample_num = self._data_loader.data_sample_num#self.data_loader.dataset.__len__()#data_sample_num#dataset.__len__()
        return self

    @staticmethod
    def make_generator(args: DatasetLoaderArgs) -> torch.Generator | None:
        """Return an independent ``torch.Generator`` seeded from ``args.generator_seed``.

        When ``generator_seed`` is ``None`` the method returns ``None``, which
        makes the DataLoader fall back to the global torch RNG (original behaviour).

        Passing a per-client seed (e.g. ``42 + client_id``) isolates each
        client's shuffle sequence from the global RNG and from other clients,
        making data ordering reproducible and independent of what other parts of
        the code do with ``torch.manual_seed``.
        """
        seed = getattr(args, "generator_seed", None)
        if seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(int(seed))
        return g

    @abstractmethod
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        Real create loader
        """
        pass

    def _get_data_length(self):
        self._try_len()
        self._try_length_hint()
        return self._count_via_loader()

    def _try_len(self):
        try:
            dl = self._data_loader
            # len(DataLoader) returns number of batches, not samples.
            # Prefer dataset length which is the true sample count.
            if hasattr(dl, "dataset") and hasattr(dl.dataset, "__len__"):
                self.data_sample_num = len(dl.dataset)
            else:
                # Fallback: batch_count * batch_size (approximate)
                batch_size = getattr(dl, "batch_size", 1) or 1
                self.data_sample_num = len(dl) * batch_size
        except Exception:
            pass

    def _try_length_hint(self):
        try:
            n = operator.length_hint(self._data_loader, -1)
            self.data_sample_num = n if n >= 0 else None
        except Exception:
            pass
        
    def _count_via_loader(self):
        total = 0
        for batch in self._data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                y = batch[1]
                if hasattr(y, "size"):      # Tensor
                    total += int(y.size(0))
                elif hasattr(y, "__len__"): # list 等
                    total += len(y)
                else:
                    total += 1
            else:
                total += 1
        self.data_sample_num = total
