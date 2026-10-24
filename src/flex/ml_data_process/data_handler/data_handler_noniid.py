import torch
from torch.utils.data import DataLoader

from .data_handler import DataHandler
from .data_handler_args import DataHandlerArgs
from ...ml_data_loader import DatasetLoaderFactory, CustomDataset
from ...ml_data_process.data_distribution import DataDistribution

class DataHandler_Noniid(DataHandler):
    def __init__(self, dataloader: DataLoader):
        """
        Args:
            dataloader (DataLoader): PyTorch DataLoader for dataset
        """
        self.dataloader = dataloader
        self.data_pool = None
        self.x_train = []
        self.y_train = []

        # Load data into memory
        self._load_data()
        self.create_data_pool()

    #override
    def _load_data(self):
        """
        Load data from DataLoader and store in x_train, y_train
        """
        images_list, labels_list = [], []
        for images, labels in self.dataloader:
            images_list.append(images)
            labels_list.append(labels)
        
        self.x_train = torch.cat(images_list, dim=0)
        self.y_train = torch.cat(labels_list, dim=0)

    #override
    def create_data_pool(self, pools = 10):
        """
        Organizes dataset into a dictionary where keys are class labels (0-9),
        and values are lists of corresponding images.

        Returns:
            dict: {label: tensor(images)}
        """
        self.data_pool = {i: [] for i in range(pools)}
        for i in range(pools):
            self.data_pool[i] = self.x_train[self.y_train.flatten() == i]

        return self.data_pool

    #override
    def generate(self, args: DataHandlerArgs):
        """
        Distributes imbalanced data to different clients based on predefined patterns and returns a list of DataLoader for each client.

        Args:
            data_volum_list (list): A list containing data volume for different classes (used only if distribution="custom").
            verify_allocate (bool): Whether to print allocation results.
            distribution (str): Default is "mnist_lt", supports different distributions.
            batch_size (int): Number of samples per batch for the DataLoader.
            shuffle (bool): Whether to shuffle the data in the DataLoader.
            num_workers (int): Number of worker threads for the DataLoader.

        Returns:
            list: A list of DataLoader objects, each corresponding to one client's data.
        """
        # Ensure data_pool is initialized
        if self.data_pool is None:
            raise ValueError("Data pool is not created. Call create_data_pool() first.")

        # Get the distribution pattern
        distribution_pattern = DataDistribution.use(args.distribution, args.data_volum_list)

        # Allocate data for each client
        allocated_data = []
        for client_idx, client_data in enumerate(distribution_pattern):
            client_images = []
            client_labels = []
            
            # Track client's distribution for verification
            client_distribution = {}
            
            # Collect data for this client from each class
            for label_idx, num_samples in enumerate(client_data):
                if num_samples > 0:
                    if num_samples > len(self.data_pool[label_idx]):
                        raise ValueError(f"Not enough samples for class {label_idx}: requested {num_samples}, available {len(self.data_pool[label_idx])}")
                    
                    # Select and remove data from pool
                    selected_data = self.data_pool[label_idx][:num_samples]
                    client_images.extend(selected_data)
                    client_labels.extend([label_idx] * num_samples)
                    self.data_pool[label_idx] = self.data_pool[label_idx][num_samples:]
                    
                    # Update distribution tracking
                    client_distribution[label_idx] = num_samples
            
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
            if args.verify_allocate:
                print(f"Client {client_idx + 1} distribution:")
                for label, count in client_distribution.items():
                    print(f"  Label {label}: {count} samples")
                print(f"  Total samples: {len(client_images)}")

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
            
            # Create DataLoader for this client
            train_loader = DatasetLoaderFactory.create_loader(
                train_dataset, 
                batch_size = args.batch_size,  # Ensure batch_size doesn't exceed dataset size
                shuffle = args.shuffle, 
                num_workers = args.num_workers)
            
            train_loaders.append(train_loader)

        return train_loaders
