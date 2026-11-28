from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

from torch.utils.data import DataLoader
from torchtext.datasets import DBpedia

'''
Dataset loader for DBpedia
'''
class DatasetLoader_DBpedia(DatasetLoader):
    def __init__(self):
        super().__init__()

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        self._dataset = DBpedia(root = args.root, split = args.split)
        self._data_loader = DataLoader(self._dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                            num_workers=args.num_workers, collate_fn = args.text_collate_fn)
        return
