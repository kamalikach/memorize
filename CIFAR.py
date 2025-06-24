import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load(**kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    transform_nn_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    nn_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_nn_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return train_dataset, nn_train_dataset, test_dataset


