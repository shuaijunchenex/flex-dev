from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_custom import CustomDataset
from torch.utils.data import DataLoader, Dataset

class DatasetLoader_Custom(DatasetLoader):
    """
    Custom dataset loader.
    """
    def __init__(self):
        super().__init__()

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        Create DataLoader(s) from a custom Dataset provided in args.dataset.
        If args.dataset is a dict from noniid_data_generator, convert it to CustomDataset.
        """
        
        # Handle dict format from noniid_data_generator {'images': [...], 'labels': [...], 'distribution': {...}}
        if isinstance(args.dataset, dict) and 'images' in args.dataset and 'labels' in args.dataset:
            self._dataset = CustomDataset(args.dataset['images'], args.dataset['labels'], transform=args.transform)
        else:
            self._dataset = args.dataset

        self.data_sample_num = len(self._dataset) 
        self.task_type = args.task_type
        generator = self.make_generator(args)

        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=args.collate_fn,
            generator=generator
        )

        return
