from typing import Any

from torch.utils.data import Dataset

class InMemoryDataset(Dataset):
    def __init__(self, other_ds: Dataset):
        self.data = list()
        self.labels = list()

        for idx in range(len(other_ds)):
            d, l = other_ds.__getitem__(idx)
            self.data.append(d)
            self.labels.append(l)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], int(self.labels[index])

    def __len__(self) -> int:
        return len(self.data)