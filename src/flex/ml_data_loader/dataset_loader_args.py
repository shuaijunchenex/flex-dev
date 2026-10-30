from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any

from ..ml_utils import KeyValueArgs, dict_exists, dict_get
from ..ml_utils.path_utils import PathUtils
from .dataset_loader_util import DatasetLoaderUtil


@dataclass
class DatasetLoaderArgs(KeyValueArgs):
    """
    Dataset loader arguments
    """
    
    # Dataset vars
    dataset_type: str = ""  # Dataset type
    root: str = ""          # data set files folder
    split: str = ""
    is_train: bool = True   # True for train, False for test
    is_download: bool = True      # is download from internet

    # Data loader vars
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = False   # set by TrainingUtils.apply_train_optimization
    # Independent RNG for DataLoader shuffle.  When set, a torch.Generator seeded
    # with this value is passed to DataLoader so that data ordering is isolated
    # from the global torch RNG.  Each client should use a different seed
    # (e.g. 42 + client_id) to get independent but reproducible shuffle sequences.
    # None means fall back to the global torch RNG (original behaviour).
    generator_seed: int | None = None

    # Collate and tramsform
    collate_fn: Any = None
    transform: Any = None
    text_collate_fn: Any = None
    vocab: Any = None
    vocab_size: Any = None
    # For custom dataset
    dataset = None

    def __init__(self, config_dict: dict|None = None, is_clone_dict = False):
        super().__init__(config_dict, is_clone_dict)

        if config_dict is not None and dict_exists(config_dict, "data_loader|dataset_loader"):
             self.set_args(dict_get(config_dict, "data_loader|dataset_loader"), is_clone_dict)

        self.dataset_type = self.get("name", "mnist")
        # Resolve root relative to the project root (lib_parent_dir) so that
        # yaml configs can use a simple path like ".dataset" instead of
        # fragile relative paths that depend on the working directory.
        raw_root = os.path.expandvars(os.path.expanduser(self.get("root", ".dataset")))
        self.root = PathUtils.resolve_path(raw_root)
        self.split = self.get("split", "")
        self.is_train = self.get("is_train", True)
        self.is_download = self.get("is_download", True)

        self.batch_size = self.get("batch_size", 64)
        self.max_len = self.get("max_len", 256)
        self.shuffle = self.get("shuffle", True)
        self.num_workers = self.get("num_workers", 4)
        self.generator_seed = self.get("generator_seed", None)
        # pin_memory: read from global train-optimization config
        from ..ml_utils.training_utils import TrainingUtils
        self.pin_memory = TrainingUtils.is_optimization_enabled("pin_memory")
        self.dataset = self.get("dataset", None)
        self.task_type = self.get("task_type", None)  # cv|nlp
        self.vocab = self.get("vocab", None)
        self.vocab_size = self.get("vocab_size", None)
        return
