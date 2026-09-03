import sys
sys.path.insert(0, '')

import numpy as np
import torch
import math
from typing import List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .noniid_distribution_generator import NoniidDistributionGenerator
from ...ml_data_loader import DatasetLoaderArgs, DatasetLoaderFactory, CustomDataset


class NoniidDataGenerator:
    def __init__(self, dataloader):
        """
        Args:
            dataloader (DataLoader): PyTorch DataLoader for dataset
        """
        self.dataloader = dataloader
        self.data_pool = None
        self.x_train = []
        self.y_train = []
        self.labels_sorted = []  # keep original label ordering for diagnostics

        # Load data into memory
        self._load_data()
        self.create_data_pool()

    # def _load_data(self):
    #     """Load data from DataLoader and store in x_train, y_train"""
    #     inputs_list, labels_list = [], []

    #     for batch in self.dataloader:
    #         if isinstance(batch, (list, tuple)) and len(batch) == 2:
    #             inputs, labels = batch
    #         else:
    #             raise ValueError(f"Unexpected batch format: {type(batch)}")

    #         inputs_list.append(inputs)
    #         labels_list.append(labels)

    #     # 图像任务：inputs 是 [B, C, H, W]；文本任务：inputs 是 [B, L]
    #     try:
    #         self.x_train = torch.cat(inputs_list, dim=0)
    #     except Exception as e:
    #         raise RuntimeError(
    #             f"Failed to concatenate inputs. "
    #             f"Check input shapes: {[x.shape for x in inputs_list]}"
    #         ) from e

    #     self.y_train = torch.cat(labels_list, dim=0)

    def _load_data(self):
        """Load data without dropping fields from HuggingFace encodings."""
        data_list, labels_list = [], []

        # Show progress bar for large datasets (e.g. CINIC-10 90k images)
        try:
            from flex.ml_utils.tqdm_utils import pbar
            dataloader_iter = pbar(self.dataloader, desc="[NoniidDataGenerator] Loading data into memory", unit="batch")
        except ImportError:
            dataloader_iter = self.dataloader

        for inputs, labels in dataloader_iter:
            # Ensure labels are tensors for downstream unique/slicing
            labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)

            # Preserve every tensor field in BatchEncoding/dict inputs. In
            # particular, sentence-pair tasks need attention_mask and
            # token_type_ids after the Non-IID split.
            if torch.is_tensor(inputs):
                norm_inputs = inputs
            elif hasattr(inputs, "items"):
                norm_inputs = {
                    key: (
                        value
                        if torch.is_tensor(value)
                        else torch.as_tensor(value)
                    )
                    for key, value in inputs.items()
                    if torch.is_tensor(value)
                    or isinstance(value, (list, tuple))
                }
                if not norm_inputs:
                    raise TypeError(
                        "Tokenizer mapping contains no tensor-compatible fields."
                    )
            elif hasattr(inputs, "ids"):
                # tokenizers.Encoding (fast) exposes .ids
                norm_inputs = torch.as_tensor(getattr(inputs, "ids"))
            else:
                try:
                    norm_inputs = torch.as_tensor(inputs)
                except Exception:
                    norm_inputs = inputs

            data_list.append(norm_inputs)
            labels_list.append(labels)

        # Merge X (data)
        if data_list and isinstance(data_list[0], dict):
            field_names = list(data_list[0])
            if any(set(batch) != set(field_names) for batch in data_list):
                raise ValueError(
                    "Tokenizer fields differ between batches; cannot split "
                    "the encoded dataset safely."
                )

            merged_fields = {}
            for key in field_names:
                values = [batch[key] for batch in data_list]
                if values[0].dim() == 2:
                    max_len = max(value.shape[1] for value in values)
                    pad_value = 0
                    if key == "input_ids":
                        for batch in data_list:
                            mask = batch.get("attention_mask")
                            ids = batch.get("input_ids")
                            if (
                                mask is not None
                                and ids is not None
                                and mask.shape == ids.shape
                            ):
                                padded = ids[mask == 0]
                                if padded.numel() > 0:
                                    pad_value = int(padded.flatten()[0].item())
                                    break
                    values = [
                        torch.nn.functional.pad(
                            value,
                            (0, max_len - value.shape[1]),
                            value=pad_value if key == "input_ids" else 0,
                        )
                        if value.shape[1] < max_len
                        else value
                        for value in values
                    ]
                merged_fields[key] = torch.cat(values, dim=0)

            sample_count = sum(
                int(labels.numel()) for labels in labels_list
            )
            inconsistent_fields = {
                key: int(value.shape[0])
                for key, value in merged_fields.items()
                if value.shape[0] != sample_count
            }
            if inconsistent_fields:
                raise ValueError(
                    "Tokenizer output fields and labels contain different "
                    f"numbers of samples: labels={sample_count}, "
                    f"fields={inconsistent_fields}"
                )

            self.x_train = [
                {
                    key: value[index]
                    for key, value in merged_fields.items()
                }
                for index in range(sample_count)
            ]
        elif data_list and torch.is_tensor(data_list[0]):
            # If sequence lengths differ across batches (common in NLP), pad to max length first
            shapes = [tuple(t.shape) for t in data_list]
            if len(set(shapes)) == 1:
                self.x_train = torch.cat(data_list, dim=0)
            else:
                # Handle 2D tensors shaped [B, L]; pad to max L
                max_len = max(s[1] for s in shapes if len(s) >= 2)
                padded_list = []
                for t in data_list:
                    if t.dim() == 2 and t.shape[1] < max_len:
                        pad_len = max_len - t.shape[1]
                        # pad on the right: (pad_right, pad_left) order per dimension
                        t = torch.nn.functional.pad(t, (0, pad_len), value=0)
                    padded_list.append(t)
                self.x_train = torch.cat(padded_list, dim=0)
        else:
            # Text task: merge into a single large list or object
            self.x_train = []
            for b in data_list:
                if isinstance(b, (list, tuple, torch.Tensor)):
                    self.x_train.extend(list(b) if torch.is_tensor(b) else b)
                else:
                    self.x_train.append(b)

        # Merge Y (labels)
        self.y_train = torch.cat(labels_list, dim=0)

    # def _load_data(self):
    #     """Load data from DataLoader and store in x_train, y_train"""
    #     images_list, labels_list = [], []
    #     for images, labels in self.dataloader:
    #         images_list.append(images)
    #         labels_list.append(labels)
        
    #     self.x_train = torch.cat(images_list, dim=0)
    #     self.y_train = torch.cat(labels_list, dim=0)

    def create_data_pool(self):
        """
        Build {class_idx: tensor(images)} dynamically based on unique labels in the dataset.
        Works for binary IMDb or 10-class MNIST alike.
        """
        uniq = torch.unique(self.y_train).tolist()
        uniq_sorted = sorted(int(l) for l in uniq)        # e.g. IMDb: [0,1]
        self.labels_sorted = uniq_sorted
        self.num_classes = len(uniq_sorted)

        # Use actual label values as keys in data_pool to avoid mismatch
        # between label indices and dataset label values (e.g. datasets
        # that use labels like [1] only). This makes lookups like
        # self.data_pool[label_idx] valid when label_idx represents the
        # actual class label from distribution matrices.
        self.data_pool = {lab: [] for lab in uniq_sorted}
        for lab in uniq_sorted:
            mask = self.y_train.flatten() == lab
            if torch.is_tensor(self.x_train):
                self.data_pool[lab] = self.x_train[mask]
            else:
                indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                self.data_pool[lab] = [self.x_train[index] for index in indices]

        return self.data_pool

    @staticmethod
    def distribution_generator(distribution='mnist_lt', data_volum_list=None):
        """
        Generates the distribution pattern for data allocation.

        Args:
            distribution (str): Type of distribution ('mnist_lt' for long-tail, 'custom' for user-defined).
            data_volum_list (list): Custom data volume distribution, required if distribution='custom'.

        Returns:
            list: A nested list where each sublist represents the data volume per class for a client.
        """
        mnist_data_volum_list_lt = [
            [592, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [592, 749, 0, 0, 0, 0, 0, 0, 0, 0],
            [592, 749, 744, 0, 0, 0, 0, 0, 0, 0],
            [592, 749, 744, 875, 0, 0, 0, 0, 0, 0],
            [592, 749, 745, 876, 973, 0, 0, 0, 0, 0],
            [592, 749, 745, 876, 973, 1084, 0, 0, 0, 0],
            [592, 749, 745, 876, 974, 1084, 1479, 0, 0, 0],
            [593, 749, 745, 876, 974, 1084, 1479, 2088, 0, 0],
            [593, 749, 745, 876, 974, 1084, 1480, 2088, 2925, 0],
            [593, 750, 745, 876, 974, 1085, 1480, 2089, 2926, 5949]
        ]

        mnist_data_volum_list_one_label = [[5920, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 6742, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 5958, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 6131, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 5842, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 5421, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 5918, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 6265, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 5851, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 5949]]

        mnist_data_volum_balance =  [[592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [593, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [593, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [593, 674, 595, 613, 584, 542, 591, 626, 585, 594]]

        cifar_data_volum_list_one_label = [[5000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 5000, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 5000, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 5000, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 5000, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 5000, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 5000, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 5000, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 5000, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 5000]]

        fmnist_data_volum_list_one_label = [[6000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 6000, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 6000, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 6000, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 6000, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 6000, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 6000, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 6000, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 6000, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 6000]]


        kmnist_data_volum_list_one_label = [[6000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 6000, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 6000, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 6000, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 6000, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 6000, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 6000, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 6000, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 6000, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 6000]]


        qmnist_data_volum_list_one_label = [[10895, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 12398, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 10952, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 11205, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 10640, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 9983, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 10917, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 11468, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 10767, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 10775]]


        imdb_two_clients_one_label = [[12500, 0],
                                      [0, 12500]]

        if distribution == "mnist_lt":
            return mnist_data_volum_list_lt
        if distribution == 'mnist_feature_shift':
            return mnist_data_volum_balance
        if distribution == 'mnist_one_label':
            return mnist_data_volum_list_one_label
        if distribution == 'cifar10_one_label':
            return cifar_data_volum_list_one_label 
        if distribution == 'imdb_two_clients_one_label':
            return imdb_two_clients_one_label
        if distribution == 'fmnist_one_label':
            return fmnist_data_volum_list_one_label
        if distribution == 'kmnist_one_label':
            return kmnist_data_volum_list_one_label
        if distribution == 'qmnist_one_label':
            return qmnist_data_volum_list_one_label
        elif distribution == "custom":
            if data_volum_list is None:
                raise ValueError("Custom distribution requires 'data_volum_list'.")
            return data_volum_list
        else:
            raise ValueError("Invalid distribution type. Choose 'mnist_lt' or 'custom'.")

    def generate_noniid_data(
        self,
        data_volum_list=None,
        verify_allocate=True,
        distribution="mnist_lt",
        batch_size=64,
        shuffle=False,
        num_workers=0,
        distribution_config: dict | None = None,
    ):
        """
        Distributes imbalanced data to different clients based on predefined patterns and returns a list of DataLoader for each client.

        Args:
            data_volum_list (list): Custom distribution matrix (used if distribution="custom").
            verify_allocate (bool): Whether to print allocation results.
            distribution (str): Default "mnist_lt"; overridden by distribution_config if provided.
            batch_size (int): Number of samples per batch for the DataLoader.
            shuffle (bool): Whether to shuffle the data in the DataLoader.
            num_workers (int): Number of worker threads for the DataLoader.
            distribution_config (dict|None): Full config dict from YAML `data_distribution` section. If provided,
                - uses `distribution_config.get("use")` to override distribution name
                - if `use == "custom"`, pulls matrix from `distribution_config["custom_define"]["custom"]` (or `data_volum_list`)

        Returns:
            list: A list of DataLoader objects, each corresponding to one client's data.
        """
        # Ensure data_pool is initialized
        if self.data_pool is None:
            raise ValueError("Data pool is not created. Call create_data_pool() first.")

        # If full config is provided, override distribution + custom matrix
        if distribution_config is not None:
            dist_name = distribution_config.get("use", distribution)
            distribution = dist_name
            custom_define = distribution_config.get("custom_define")
            if dist_name == "custom":
                custom = None
                if isinstance(custom_define, dict):
                    custom = custom_define.get("custom")
                # fallback to explicit argument if present
                if custom is not None:
                    data_volum_list = custom
            elif isinstance(custom_define, dict) and dist_name in custom_define:
                # Named custom distribution (e.g. cifar10_dirichlet_0_4)
                data_volum_list = custom_define[dist_name]
                distribution = "custom"

        # Always use the provided distribution matrix (predefined or custom).
        distribution_pattern = self.distribution_generator(distribution, data_volum_list)

        # Allocate data for each client
        allocated_data = []
        for client_idx, client_data in enumerate(distribution_pattern):
            client_images = []
            client_labels = []
            
            # Track client's distribution for verification
            client_distribution = {}
            
            # Collect data for this client from each class
            for col_idx, num_samples in enumerate(client_data):
                if num_samples > 0:
                    # Map the column index in the distribution matrix to the actual label value
                    # (e.g., if labels_sorted is [1, 2], then col_idx 0 maps to label 1)
                    if col_idx < len(self.labels_sorted):
                        label_val = self.labels_sorted[col_idx]
                    else:
                        label_val = col_idx # Fallback if matrix cols > available labels
                    
                    pool = self.data_pool.get(label_val)
                    if pool is None:
                        raise ValueError(
                            f"Label {label_val} (from matrix index {col_idx}) not in data pool. Available: {self.labels_sorted}"
                        )
                    available = len(pool)
                    if num_samples > available:
                        from ...ml_utils import console
                        console.warn(
                            f"Not enough samples for class {label_val}: requested {num_samples}, available {available}; capping to {available}"
                        )
                        num_samples = available
                    if num_samples == 0:
                        continue

                    # Select and remove data from pool
                    selected_data = pool[:num_samples]
                    client_images.extend(selected_data)
                    client_labels.extend([label_val] * num_samples)
                    self.data_pool[label_val] = pool[num_samples:]
                    
                    # Update distribution tracking
                    client_distribution[col_idx] = num_samples
            
            # Skip clients with no data
            if len(client_images) == 0:
                continue
                
            # Store client data
            allocated_data.append({
                'images': client_images,
                'labels': client_labels,
                'distribution': client_distribution
            })
            
            # Verify allocation results
            if verify_allocate:
                from ...ml_utils import console
                console.debug(f"Client {client_idx + 1} distribution:")
                for label, count in client_distribution.items():
                    console.debug(f"  Label {label}: {count} samples")
                console.debug(f"  Total samples: {len(client_images)}")

        # Create DataLoader for each client
        train_loaders = []
        
        for client_idx, client_data in enumerate(allocated_data):
            # Skip if client has no data
            if len(client_data['images']) == 0:
                continue
            
            # Create dataset (without transform)
            train_dataset = CustomDataset(
                client_data['images'], 
                client_data['labels'], 
                transform=None  # No transform applied
            )

            train_loaders.append(train_dataset)

        return train_loaders
    
        
    # def generate_noniid_data(self, data_volum_list=None, verify_allocate=True,
    #                         distribution="mnist_lt", batch_size=64, shuffle=False, num_workers=0):
    #     """
    #     Distributes imbalanced data to different clients and returns a list of DataLoader.
    #     Each returned DataLoader instance will have extra attributes:
    #         - length: total number of samples for this client
    #         - num_batches: total number of batches for this client
    #         - class_distribution: dict {label_idx: count}
    #     """
    #     # Ensure data_pool is initialized
    #     if self.data_pool is None:
    #         raise ValueError("Data pool is not created. Call create_data_pool() first.")

    #     # Get the distribution pattern (二维表：行=client，列=类别)
    #     distribution_pattern = self.distribution_generator(distribution, data_volum_list)

    #     # Allocate data for each client
    #     allocated_data = []
    #     for client_idx, client_row in enumerate(distribution_pattern):
    #         client_images = []
    #         client_labels = []

    #         # Track this client's distribution
    #         client_distribution = {}

    #         for label_idx, num_samples in enumerate(client_row):
    #             if num_samples <= 0:
    #                 continue
    #             if num_samples > len(self.data_pool[label_idx]):
    #                 raise ValueError(
    #                     f"Not enough samples for class {label_idx}: "
    #                     f"requested {num_samples}, available {len(self.data_pool[label_idx])}"
    #                 )

    #             # Select and remove data from pool
    #             selected_data = self.data_pool[label_idx][:num_samples]
    #             client_images.extend(selected_data)
    #             client_labels.extend([label_idx] * num_samples)
    #             self.data_pool[label_idx] = self.data_pool[label_idx][num_samples:]

    #             client_distribution[label_idx] = num_samples

    #         if len(client_images) == 0:
    #             continue

    #         allocated_data.append({
    #             "images": client_images,
    #             "labels": client_labels,
    #             "distribution": client_distribution
    #         })

    #         if verify_allocate:
    #             print(f"Client {client_idx + 1} distribution: {client_distribution} | Total: {len(client_images)}")

    #     # Create DataLoader for each client and attach `length` etc.
    #     train_loaders = []

    #     for client_idx, client_data in enumerate(allocated_data):
    #         if len(client_data["images"]) == 0:
    #             continue

    #         # Create dataset
    #         train_dataset = CustomDataset(
    #             client_data["images"],
    #             client_data["labels"],
    #             transform=None  # No transform applied here;按需替换
    #         )

    #         loader = CustomDataset.create_custom_loader(
    #             train_dataset,
    #             batch_size=batch_size,
    #             shuffle=shuffle,
    #             num_workers=num_workers,
    #             collate_fn=None
    #         )

    #         total_samples = len(train_dataset)
    #         loader.data_sample_num = total_samples                    # 样本总数
    #         loader.num_batches = math.ceil(total_samples / max(1, batch_size))  # 批次数
    #         loader.class_distribution = client_data["distribution"]             # 类别分布（可选）

    #         train_loaders.append(loader)

    #     return train_loaders
