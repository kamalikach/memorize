import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load(**kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image or numpy array to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
        ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    nn_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    return train_dataset, nn_train_dataset, test_dataset

