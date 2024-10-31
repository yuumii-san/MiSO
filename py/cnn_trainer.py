#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a CNN model and obtain uStim reponse predictions to all possible uStim patterns.

This script trains a CNN model using the merged training dataset across sessions
and use the trained model to obtain the uStim response predictions.

Author: Yuki Minai
Created: 2024-11-01
Version: 1.0.0
License: CC BY-NC-ND 4.0
"""

import os

import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GridDataset(Dataset):
    """
    Grid format input dataset class for a CNN model.
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)  # Convert input to tensor
        y = torch.tensor(self.outputs[idx], dtype=torch.float32) # Convert output to tensor
        return x, y
  
    
class CNN(nn.Module):
    """
    Convolutional Neural Network class.
    """
    def __init__(self, target_dims, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        if self.kernel_size==3:            
            self.dropout = nn.Dropout(0.2)
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(6*6*64, 32)
            self.fc2 = nn.Linear(32, 10)
            self.fc3 = nn.Linear(10, target_dims)    
        
            torch.nn.init.xavier_normal_(self.conv1.weight)
            torch.nn.init.xavier_normal_(self.conv2.weight)
            torch.nn.init.xavier_normal_(self.fc1.weight)
            torch.nn.init.xavier_normal_(self.fc2.weight)
            torch.nn.init.xavier_normal_(self.fc3.weight)
            
        elif self.kernel_size==2:
            self.conv1 = nn.Conv2d(1, 5, kernel_size=2, stride=1)
            self.conv2 = nn.Conv2d(5, 10, kernel_size=2, stride=1)
            self.fc1 = nn.Linear(8*8*10, 64)
            self.fc2 = nn.Linear(64, 5)
            self.fc3 = nn.Linear(5, target_dims)
            
            torch.nn.init.xavier_normal_(self.conv1.weight)
            torch.nn.init.xavier_normal_(self.conv2.weight)
            torch.nn.init.xavier_normal_(self.fc1.weight)
            torch.nn.init.xavier_normal_(self.fc2.weight)
            torch.nn.init.xavier_normal_(self.fc3.weight)
        
                                       
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # From (B,H,W,C) -> (B,C,H,W)
        if self.kernel_size==3:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.kernel_size==2:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        return x
    

def train_cnn(config_path):
    """
    Create CNN training data by using latent space alignment.

    Parameters:
        config_path (str): Path to the configuration file.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    input_path_base = config['input'].get('input_path_base', './')
    input_filename = config['input'].get('input_filename', [])
    cmp_path_base = config['input'].get('cmp_path_base', './')
    cmp_filename = config['input'].get('cmp_filename', '')
    target_dimensions_ind = config['input'].get('target_dimensions_ind', [0])
    kernel_size = config['cnn'].get('kernel_size', 3)
    train_val_test_split = config['train'].get('train_val_test_split', [])
    batch_size = config['train'].get('batch_size', 32)
    num_training_epochs = config['train'].get('num_training_epochs', [])
    lr = config['train'].get('lr', 0.001)
    optim = config['train'].get('optimizer', 'Adam')
    verbose = config['train'].get('verbose', False)
    output_path = config['output'].get('output_path', './')
    output_filename = config['output'].get('output_filename', './')

    print('Transform input stim_chan into 10x10 array format')
    cmp_map = loadmat(cmp_path_base + cmp_filename)
    ARRAY_MAP = cmp_map['arraymap']
    stim_chans_train = np.loadtxt(f"{input_path_base}{input_filename}_stim_chans.csv", delimiter=',')
    arrays = []
    for sample_elec in stim_chans_train:
        array = np.zeros((10, 10))
        array[ARRAY_MAP == sample_elec] = 1
        arrays.append(array.reshape(10, 10, 1))
    arrays = np.array(arrays)  # shape (#data, 10, 10, 1)

    print('Load aligned latent activity data')
    zs_train = np.loadtxt(f"{input_path_base}{input_filename}_zs.csv", delimiter=',')
    zs_train_filtered = zs_train[:, target_dimensions_ind]

    print('Prepare dataset and dataloader')
    # Initialize dataset
    dataset = GridDataset(arrays, zs_train_filtered)

    # Split the dataset into train, validation, and test sets (e.g. 60% train, 20% val, 20% test)
    train_size = int(train_val_test_split[0] * len(dataset))
    val_size = int(train_val_test_split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoader for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Set batch size as needed
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Fit a model')
    fit_hist_dict_all = {}
    net = CNN(len(target_dimensions_ind), kernel_size=kernel_size)
    criterion = nn.MSELoss()

    # Define an optimizer
    if optim == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # Train a CNN
    loss_val_hist = []
    min_loss_val = np.inf
    min_loss_epoch = 0
    for epoch in range(num_training_epochs):  # loop over the dataset multiple times
        net.to(device)
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs = data[0]
            inputs = inputs.to(torch.float32).to(device)
            labels = data[1]
            labels = labels.to(torch.float32).to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if verbose:
                if i % 50 == 49: # print every 50 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] training loss: {running_loss / 50:.3f}')
                    running_loss = 0.0

        # Assess validation performance
        net.eval()
        running_loss_val = 0.0
        N_valid = len(valid_dataloader)
        for j, data in enumerate(valid_dataloader):
            inputs = data[0]
            inputs = inputs.to(torch.float32).to(device) # to solve an error: Input type (double) and bias type (float) should be the same
            labels = data[1]
            labels = labels.to(torch.float32).to(device)

            outputs_val = net(inputs)
            loss_val = criterion(outputs_val, labels)
            running_loss_val += loss_val.item()
        avg_loss_val = running_loss_val / N_valid
        loss_val_hist.append(avg_loss_val)

        # Display the validation performance at every 50 epoch
        if epoch % 50 == 49:
            print(f'{epoch + 1}, validation loss {running_loss_val / N_valid:.3f}')

        # Save the best validation model so far
        if epoch==0:
            model_dir = f"{output_path}model/cnn/"

            # Create a directory if it does not exist
            isExist = os.path.exists(model_dir)
            if not isExist:
                os.makedirs(model_dir)

            model_path = f'{model_dir}{output_filename}.pth'
            torch.save(net.state_dict(), model_path) # save model 
        elif min_loss_val > avg_loss_val:
            min_loss_val = avg_loss_val
            min_loss_epoch = epoch
            torch.save(net.state_dict(), model_path) # save model

    print('Finish Training')
    
    print('Get CNN prediction to all patterns')
    # Create inputs to all patterns
    inputs_all_model = [] # format of model input
    inputs_all_model_interpretable = [] # interpretable format of each pattern
    predictions_all_model = []

    # Transform all patterns into 10x10 array format
    all_patterns = np.arange(1,97,1) # all single electrodes
    inputs = []  
    for i in all_patterns:
        input_elec = np.zeros((10,10))
        input_elec[ARRAY_MAP==i]=1
        input_elec = input_elec.reshape([1, 10, 10, 1])
        inputs.append(input_elec)
            
    # Get prediction using the trained model
    predictions = []
    for input_elec in inputs:
        net.eval()
        output = net(torch.tensor(input_elec, dtype=torch.float32).to(device))
        predictions.append(output.cpu().detach().numpy().flatten())

    print(f"Saving CNN prediction data to {output_path}output/cnn_prediction/prediction_{output_filename}.csv")
    predictions = np.array(predictions)
    np.savetxt(f"{output_path}output/cnn_prediction/prediction_{output_filename}.csv", predictions, delimiter=',', fmt='%s') 
