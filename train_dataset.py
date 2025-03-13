import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader

from cifar.dataset import get10
from cifar.model import cifar10
from mnist.dataset import get as get_mnist
from mnist.model import mnist
from svhn.dataset import get as get_svhn
from svhn.model import svhn


def train_cfar(sample_func: Callable[[np.ndarray[float]], list[int]],
               epochs: int = 100,
               test_interval: int = 5,
               seed: int = None,
               early_stopping_min_epochs: int = 30,
               early_stopping_patience: int = 10
               ) -> tuple[nn.Module, list[float], list[float]]:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    train_loader, test_loader = get10(32, sample_func=sample_func)
    cifar_model = cifar10(128)
    return train_dataset(cifar_model, train_loader, test_loader, learning_rate=1e-3, weight_decay=0.0,
                         epochs=epochs, test_interval=test_interval,
                         early_stopping_min_epochs=early_stopping_min_epochs,
                         early_stopping_patience=early_stopping_patience)

def train_mnist(sample_func: Callable[[np.ndarray[float]], list[int]],
                epochs: int = 100,
                test_interval: int = 5,
                seed: int = None,
                early_stopping_min_epochs: int = 40,
                early_stopping_patience: int = 10) -> tuple[nn.Module, list[float], list[float]]:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    train_loader, test_loader = get_mnist(16, sample_func=sample_func)
    mnist_model = mnist(input_dims=784, n_hiddens=[256, 256], n_class=10)
    return train_dataset(mnist_model, train_loader, test_loader, learning_rate=1e-3, weight_decay=0.0001,
                         epochs=epochs, test_interval=test_interval,
                         early_stopping_min_epochs=early_stopping_min_epochs,
                         early_stopping_patience=early_stopping_patience)

def train_svhn(sample_func: Callable[[np.ndarray[float]], list[int]],
               epochs: int = 100,
               test_interval: int = 5,
               seed: int = None,
               early_stopping_min_epochs: int = 30,
               early_stopping_patience: int = 10) -> tuple[nn.Module, list[float], list[float]]:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    train_loader, test_loader = get_svhn(16, sample_func=sample_func)
    svhn_model = svhn(32)
    return train_dataset(svhn_model, train_loader, test_loader, learning_rate=1e-3, weight_decay=0.001,
                         epochs=epochs, test_interval=test_interval,
                         early_stopping_min_epochs=early_stopping_min_epochs,
                         early_stopping_patience=early_stopping_patience)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_dataset(model: nn.Module,
                  train_loader: DataLoader, test_loader: DataLoader,
                  learning_rate: float,
                  weight_decay: float,
                  epochs: int = 100,
                  test_interval: int = 5,
                  early_stopping_min_epochs: int = 30,
                  early_stopping_patience: int = 10) -> tuple[nn.Module, list[float], list[float]]:
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if torch.cuda.is_available():
        model.cuda()

    t_begin = time.time()
    accuracies = list()
    losses = list()
    stopper = EarlyStopper(early_stopping_patience)
    for epoch in range(epochs):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * epochs - elapse_time
        print(f"Epoch #{epoch + 1} Elapsed {elapse_time:.2f}s, {speed_epoch:.2f} s/epoch, {speed_batch:.2f} s/batch, ets {eta:.2f}s")

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                indx_target = target.clone()
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # data, target = Variable(data, volatile=True), Variable(target)
                output = model(data)
                test_loss += cross_entropy(output, target).item()
                _, max_idx_class = output.max(dim=1)
                correct += max_idx_class.cpu().eq(indx_target).sum().item()

            test_loss = test_loss / len(test_loader)  # average over number of mini-batch
            acc = 100. * correct / len(test_loader.dataset)
        accuracies.append(acc)
        losses.append(test_loss)
        if test_interval and (epoch % test_interval == 0 or epoch == epochs - 1):
            print(f'\tTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)')

        if stopper.early_stop(test_loss) and epoch >= early_stopping_min_epochs:
            print("Stopped early!")
            print(f'\tTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)')
            break

    return model, accuracies, losses


