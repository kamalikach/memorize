import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from utils import *
import importlib
import yaml
import argparse
import os


def corrupt_labels(dataset, ratio, classes=10, seed=0):
    torch.manual_seed(seed)
    labels = torch.tensor([y for (x, y) in dataset])
    n_corrupt = int(len(labels) * ratio)
    indices_to_replace = torch.randperm(len(labels))[:n_corrupt]
    labels[indices_to_replace] = torch.randint(low=0, high=10, size=(n_corrupt,))
    return labels

def save_modified_dataset(new_train_labels, file_prefix):
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)
    torch.save(new_train_labels, file_prefix + 'train_labels.pt')
    print('Saved train and test data\n')


def main(dataset_name, p):
    file_prefix = './data/' + dataset_name + str(p) + '/'
    print('Dataset, Saving in:', dataset_name, file_prefix)
    
    # load the full train and test datasets
    data_module = importlib.import_module(dataset_name)
    train_dataset, nn_train_dataset, test_dataset = data_module.load()
    
    new_labels = corrupt_labels(train_dataset, p)
    save_modified_dataset(new_labels, file_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Corrupt labels for a dataset')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--p', type=float, required=True, help='Corruption probability')
    args = parser.parse_args()
    main(args.dataset_name, args.p)
