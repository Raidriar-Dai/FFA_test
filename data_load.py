import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        # change image to tensor;
        ToTensor(),
        # Normalize the image by subtracting a known mean and standard deviation;
        Normalize((0.1307,), (0.3081,)),
        # flatten the image;
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def CIFAR_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        # change image to tensor;
        ToTensor(),
        # Normalize the image by subtracting a known mean and standard deviation;
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        # flatten the image;
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        CIFAR10('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        CIFAR10('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


class Labeled_Dataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.data = x
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

