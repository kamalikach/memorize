import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load(**kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize RGB channels
            ])

    transform_nn_train = transform

    train_dataset = datasets.SVHN(root='./data', split="train", download=True, transform=transform)
    nn_train_dataset = datasets.SVHN(root='./data', split="train", download=True, transform=transform_nn_train)
    test_dataset = datasets.SVHN(root='./data', split="test", download=True, transform=transform)

    return train_dataset, nn_train_dataset, test_dataset

