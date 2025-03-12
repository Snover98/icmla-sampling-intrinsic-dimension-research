from typing import Callable

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os


def get(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True,
        sample_func: Callable[[np.ndarray[float]], np.ndarray[int]] = None, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        return int(target) - 1

    ds = []
    if train:
        train_dataset = datasets.SVHN(root=data_root, split='train', download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                      target_transform=target_transform, )
        if sample_func is not None:
            train_dataset = Subset(train_dataset, sample_func(train_dataset.data))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                target_transform=target_transform
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
