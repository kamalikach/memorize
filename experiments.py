import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from utils import *
import importlib
import yaml
import os
import argparse


def bayes_model_train_and_pred(config, file_prefix, bayes_nn_train_dataset, test_dataset, bayes_test_dataset, device):
    # Now train the Bayes model, save it, save predictions on LOO dataset
    class_name = config['dataset']+'_'+config['bayes_model']
    cfg = config['training']
    
    bayes_model, test_accuracy = model_train_and_test_wrapper(class_name, cfg, bayes_nn_train_dataset, test_dataset, device)
    torch.save(bayes_model.state_dict(), file_prefix+'bayes_model.pt')
    
    bayes_preds = get_model_prediction_list(bayes_model, bayes_test_dataset, device)
    print(bayes_preds)
    torch.save(bayes_preds, file_prefix+'bayes_preds.pt')


def main(config_file):
    
    config = load_config(config_file)
    print('Main Experiment Config:', config)
    file_prefix = './data/'+config['directory']+'_'+str(config['N'])+'/'
    print('Saving in directory:', file_prefix)
    
    # Get GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the full train and test datasets
    data_module = importlib.import_module(config['loader'])
    train_dataset, nn_train_dataset, test_dataset = data_module.load(dataset=config['dataset'], directory='./data/' + config['directory'])
    
    # Generate the subsets (for LOO experiments) and training and test sets for the Bayes model
    indices = torch.randperm(len(train_dataset))[:config['N']]
    train_dataset_subset = torch.utils.data.Subset(train_dataset, indices)
    nn_train_dataset_subset = torch.utils.data.Subset(nn_train_dataset, indices)
    ########save the indices to disk
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)
    torch.save(indices, file_prefix + 'indices.pt')
    
    remove_indices = indices[:config['K']]
    all_indices = torch.arange(len(train_dataset))
    mask = ~torch.isin(all_indices, remove_indices)
    bayes_train_dataset = Subset(train_dataset, all_indices[mask])
    bayes_nn_train_dataset = Subset(nn_train_dataset, all_indices[mask])
    bayes_test_dataset = Subset(train_dataset, remove_indices)
    
    # Sanity checks
    print(len(train_dataset), len(bayes_train_dataset), len(bayes_nn_train_dataset), len(bayes_test_dataset))

    bayes_model_train_and_pred(config, file_prefix, bayes_nn_train_dataset, test_dataset, bayes_test_dataset, device)    
    
    # Now get ready for the LOO experiments
    model_configs = []
    for model_path in config["models"]:
        cfg = load_config(model_path)
        model_configs.append(cfg)
    print(model_configs)

    for experiment in model_configs:
        # Need to build class_name, expt_cfg, cfg
        class_name = config['dataset']+'_'+experiment['model']
        cfg = experiment['training']
        expt_cfg = { 'N': config['N'], 'K': config['K'], 'file_prefix': file_prefix, 'model_prefix': experiment['model'] }
        print('class_name:' , class_name)
        print('cfg:', cfg )
        print('expt_cfg:', expt_cfg)
        
        loo_preds = loo_experiment_wrapper(class_name, cfg, expt_cfg, nn_train_dataset_subset, train_dataset_subset, test_dataset, device)
        print(loo_preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My Script')
    parser.add_argument('config_file', type=str, help='Path to config file')
    args = parser.parse_args()
    main(args.config_file)
