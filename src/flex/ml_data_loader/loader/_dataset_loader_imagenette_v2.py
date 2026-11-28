
from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..image_benchmark_data import ImageBenchmarkDataset, image_transform, prepare_dataset, read_manifest


class DatasetLoader_ImagenetteV2(DatasetLoader):
    dataset_name = "imagenette_v2"

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """Use the Imagenette validation split for round and final evaluation."""
        manifest_path = args.get("manifest_path")
        if not manifest_path:
            raise ValueError(f"{self.dataset_name} requires data_loader.manifest_path")
        catalog = prepare_dataset(self.dataset_name, args.root, download=args.is_download)
        splits = read_manifest(manifest_path, catalog)["splits"]

        eval_transform = getattr(args, "test_transform", None) or image_transform(self.dataset_name, False)
        self._test_dataset = ImageBenchmarkDataset(catalog, splits["monitor"], eval_transform)
        self.final_dataset = self._test_dataset
        if splits["final"] != splits["monitor"]:
            self.final_dataset = ImageBenchmarkDataset(catalog, splits["final"], eval_transform)

        self._dataset = self._test_dataset
        if args.is_train:
            train_transform = args.transform or image_transform(self.dataset_name, True)
            self._dataset = ImageBenchmarkDataset(catalog, splits["train"], train_transform)

        self.task_type = "cv"
        self.data_sample_num = len(self._dataset)
        self._data_loader = DatasetLoader.create_loader(
            self._dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=args.is_train and args.shuffle, generator=self.make_generator(args),
            pin_memory=args.pin_memory,
        )
        self._test_data_loader = DatasetLoader.create_loader(
            self._test_dataset, batch_size=args.get("test_batch_size", args.batch_size),
            num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_memory,
        )
