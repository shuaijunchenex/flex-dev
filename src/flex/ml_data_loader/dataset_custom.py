from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @staticmethod
    def create_custom_loader(dataset, batch_size, shuffle=True, num_workers=4, collate_fn=None):
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset: The dataset to load data from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            num_workers (int): Number of worker threads.
            collate_fn (callable, optional): Function to merge a list of samples into a batch.

        Returns:
            DataLoader: A PyTorch DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )