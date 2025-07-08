import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import importlib
from itertools import product

import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

###########data analysis functions

def print_statistics(correct):
    print('Bayes :', count_rows_matching_pattern(correct, [ True, '*', '*' ]))
    print('f(S-x):', count_rows_matching_pattern(correct, [ '*', True, '*' ]))
    print('f(S  ):', count_rows_matching_pattern(correct, [ '*', '*', True ]))

    options = [True, False]

    for combo in product(options, repeat=3):
        print(list(combo), count_rows_matching_pattern(correct, list(combo)))

def analyze_data(dataset_name, directory, model, K):
    #first load the various statistics and real labels
    bayes = torch.load(directory+'bayes_preds.pt')
    loo = torch.load(directory+model+'_loopreds.pt')

    data_module = importlib.import_module(dataset_name)
    train_dataset, nn_train_dataset, test_dataset = data_module.load(dataset=dataset_name, directory='./data/'+dataset_name+'/')
    indices = torch.load(directory+'indices.pt')[0:K]
    real_labels = [train_dataset[i][1] for i in indices]

    #now calculate the two combined tables -- one wrt noisy labels, other real labels
    bayes_correct = [ [ a[0] == a[1] ] for a in bayes ]
    loo_correct = [ [ a[0] == a[1], a[0] == a[2]] for a in loo ]
    combined_correct = torch.tensor([ ai + bi for ai, bi in zip(bayes_correct, loo_correct) ])
    
    ######another way: pretend bayes predictions are the real labels
    bayes_correct_real = [ [ a[0] == b ] for a, b in zip(bayes, real_labels) ]
    combined_correct_real = torch.tensor( [ ai + bi for ai, bi in zip(bayes_correct_real, loo_correct) ] )

    return combined_correct, combined_correct_real

def count_rows_matching_pattern(A, pattern):
    ##args: A (torch.Tensor): Boolean tensor of shape [n, 3].
    #pattern (list): List of length 3 with elements True, False, or '*'.
 
    if A.shape[1] != len(pattern):
        raise ValueError("Pattern length must match number of columns in A")
    mask = torch.ones(A.size(0), dtype=torch.bool)  # start with all True
    for col_idx, pat_val in enumerate(pattern):
        if pat_val == '*':
            continue  # wildcard, ignore this column
        else:
            mask = mask & (A[:, col_idx] == pat_val)

    return mask.sum().item()


##############basic training and test functions

def model_train(model, cfg, train_loader, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    for epoch in range(cfg['epochs']):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{cfg['epochs']}], Loss: {loss.item():.4f}")
    return model

def find_accuracy(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return correct/total

def get_model_prediction(model, image, device='cpu'):
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
    predicted_label = output.argmax(dim=1).item()
    return predicted_label

def get_model_prediction_list(model, test_dataset, device='cpu'):
    data_loader = DataLoader(test_dataset, batch_size = 1)
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.append([labels.item(), predicted.item()])

    return predictions


def model_train_loo(model, cfg, train_dataset, loo_index, device='cpu'):
    print(loo_index)
    indices = list(range(0, loo_index)) + list(range(loo_index+1, len(train_dataset)))
    train_dataset_subset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = DataLoader(train_dataset_subset, batch_size=cfg['batch_size'], shuffle=True)

    model = model_train(model, cfg, train_loader, device) 
    return model


def model_train_and_test_wrapper(model_class_name, cfg, train_dataset, test_dataset, device='cpu'):
    #initialization before model training
    model = globals()[model_class_name]().to(device) 
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=cfg['batch_size'])
    
    model = model_train(model, cfg, train_loader, device)
    test_accuracy = find_accuracy(model, test_loader, device)
    return model, test_accuracy

def loo_experiment_wrapper(model_class_name, cfg, expt_cfg, train_dataset, train_dataset_nontransformed, test_dataset, device):
    
    model = globals()[model_class_name]().to(device) 

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=cfg['batch_size'])
    
    #now train model without LOO
    model = model_train(model, cfg, train_loader, device)
    test_accuracy = find_accuracy(model, test_loader, device)
    prefix = expt_cfg['file_prefix'] + expt_cfg['model_prefix']
    torch.save(model.state_dict(), prefix + '_model.pt')

    loo_preds = []

    for loo_index in range(expt_cfg['K']):
        model_t = globals()[model_class_name]().to(device) 
        
        model_t = model_train_loo(model_t, cfg, train_dataset, loo_index, device)
        test_accuracy = find_accuracy(model_t, test_loader, device)
        torch.save(model_t.state_dict(), prefix + '_model_loo_' + str(loo_index) + '.pt')

        #when evaluating on the loo example, use the non-augmented version
        image, label = train_dataset_nontransformed[loo_index]
        predicted_label_loo = get_model_prediction(model_t, image, device)
        predicted_label = get_model_prediction(model, image, device)

        loo_preds.append([label, predicted_label_loo, predicted_label])
        print(label, predicted_label_loo, predicted_label)

    torch.save(loo_preds, prefix+'_loopreds.pt')
    return loo_preds
    
############# Model classes for different datasets

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),             # 28x28 -> 14x14
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, 1, 1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),             # 14x14 -> 7x7
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class MNIST_Linear(nn.Module):
        def __init__(self):
            super(MNIST_Linear, self).__init__()
            self.layers=nn.Linear(28*28, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)   #flatten out the tensor
            x = self.layers(x)
            return x


class CIFAR_Linear(nn.Module):
        def __init__(self):
            super(CIFAR_Linear, self).__init__()
            self.layers=nn.Linear(3*32*32, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # flatten
            x = self.layers(x)
            return x

class CIFAR_Resnet(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(CIFAR_Resnet, self).__init__()
        
        # Load standard ResNet18
        self.model = resnet18(weights=None)
        # Modify the first conv layer to suit CIFAR-10 (3x32x32 input)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the first maxpool layer (not needed for small images)
        self.model.maxpool = nn.Identity()
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class SVHN_Linear(nn.Module):
        def __init__(self):
            super(SVHN_Linear, self).__init__()
            self.layers=nn.Linear(3*32*32, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # flatten
            x = self.layers(x)
            return x

class SVHN_CNN(nn.Module):
    def __init__(self):
        super(SVHN_CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x3 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x16x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 8x8x128

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 4x4x256
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SVHN_VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_VGG, self).__init__()

        # Feature extractor: reduced VGG-style layers for small input (32x32)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 16x16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 8x8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 4x4
        )

        # Classifier: matches smaller feature size
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
