from typing import Callable

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

from in_memory_dataset import InMemoryDataset


def get10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True,
          sample_func: Callable[[np.ndarray[float]], list[int]] = None, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.Compose(
            [transforms.Pad(4), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
        if sample_func is not None:
            greyscale_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.Compose(
                [transforms.Grayscale(num_output_channels=1)]))
            train_dataset = Subset(train_dataset, sample_func(np.stack([greyscale_dataset[idx][0] for idx in range(len(greyscale_dataset))])))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            InMemoryDataset(datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

