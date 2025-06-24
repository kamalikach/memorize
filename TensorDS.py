import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import importlib

def load(**kwargs):
    dataset = kwargs.get('dataset')
    if not dataset:
        raise ValueError('dataset parameter is required')

    directory = kwargs.get('directory')
    if not directory:
        raise ValueError("directory parameter is required")

    data_module = importlib.import_module(dataset)
    train_dataset, nn_train_dataset, test_dataset = data_module.load()

    new_labels = torch.load(directory + '/train_labels.pt')

    if hasattr(train_dataset, 'targets'):
        train_dataset.targets = new_labels
        nn_train_dataset.targets = new_labels
    elif hasattr(train_dataset, 'labels'):
        train_dataset.labels = new_labels
        nn_train_dataset.labels = new_labels
    else:
        raise AttributeError("Dataset has neither 'targets' nor 'labels'.")

    return train_dataset, nn_train_dataset, test_dataset

