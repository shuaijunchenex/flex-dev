from __future__ import annotations

from torch.utils.data import DataLoader
from torchtext.datasets import YahooAnswers

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

'''
Dataset loader for YahooAnswers
'''

class DatasetLoader_YahooAnswers(DatasetLoader):
    def __init__(self):
        super().__init__()

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        self._dataset = YahooAnswers(root = args.root, split = args.split)
        self._data_loader = DataLoader(self._dataset, 
                                       batch_size = args.batch_size, 
                                       shuffle = args.shuffle, 
                                       num_workers = args.num_workers, 
                                       collate_fn = args.text_collate_fn)
        return
