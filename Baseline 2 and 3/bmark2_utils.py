import csv
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import random
from sklearn.datasets import load_digits
from os import listdir
import re
from PIL import Image
from torchvision import transforms


def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for X, Y in data_loader:
            X = X.view(-1, 22)
            logits = net.forward(X)
            pred_Y = torch.argmax(logits, 1)
            num_examples += Y.size(0)
            correct_pred += (pred_Y == Y).sum()
        return correct_pred.float()/num_examples * 100

def epoch_loss(net, loss_func, data_loader):
    ''' Computes the loss of the model on the entire dataset given by the dataloader.
    Be sure to wrap this function call in a torch.no_grad statement to prevent
    gradient computation.

    @param net: The neural network to be evaluated
    @param loss_func: The loss function used to evaluate the neural network
    @param data_loader: The DataLoader which loads minibatches of the dataset

    @return The network's average loss over the dataset.
    '''
    total_examples = 0
    losses = []
    for X, Y in data_loader:
        total_examples += len(X)
        # print(loss_func(net(X), Y).item() * len(X))
        if True in torch.isnan(X):
            X[0][4] = float(0)
            X[0][5] = float(10)
            #print(X)
        loss = (loss_func(net(X), Y).item() * len(X))
        losses.append(loss) # Compute total loss for batch
        if(loss == float('nan')):
            print('NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #print(xb)
        # print(net(X).shape)
        # print(Y.shape)
        # losses.append(loss_func(net(X), Y).item() * len(X)) # Compute total loss for batch
        # loss = 0
        # ex = net(X)
        # why = Y[0]
        # print(why.shape)
        # for i in range(6):
        #     loss += loss_func(ex, why[i])
        # print(loss)
        # losses.append(loss.item() * len(X)) # Compute total loss for batch

    return torch.tensor(losses).sum() / total_examples

def train_batch(net, loss_func, xb, yb, opt=None):
    ''' Performs a step of optimization.

    @param net: the neural network
    @param loss_func: the loss function (can be applied to model(xb), yb)
    @param xb: a batch of the training data to input to the model
    @param yb: a batch of the training labels to input to the model
    @param opt: a torch.optimizer.Optimizer used to improve the model.
    '''
    if True in torch.isnan(xb):
        xb[0][4] = float(0)
        xb[0][5] = float(10)
        #print(xb)
    nets = net(xb)
    why = yb
    #print('net', nets, nets.shape)
    #print('y', why, why.shape)
    loss = loss_func(nets, why)
    #print('loss', loss)
    if(torch.isnan(loss).item()):
        print('NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #print(xb)
    #print(torch.isnan(loss).item())
    loss.backward()
    # print('Grads', net.linear.weight.grad)
    # print('Weights', net.linear.weight)
    opt.step()
    opt.zero_grad()

def floatify_df(df, s):
    headings = {
        'N': 1,
        'E': 2,
        'S': 3,
        'W': 4,
        'NE': 5,
        'NW': 6,
        'SE': 7,
        'SW': 8,
    }
    for i in range(0,s):
        if type(df['EntryStreetName'][i]) is str:
            nameint = (float(len(df['EntryStreetName'][i])))
        else:
            nameint = (float(df['EntryStreetName'][i]))
        df['EntryStreetName'][i] = nameint
        if type(df['ExitStreetName'][i]) is str:
            nameint = (float(len(df['ExitStreetName'][i])))
        else:
            nameint = (float(df['ExitStreetName'][i]))
        df['ExitStreetName'][i] = nameint
        if type(df['Path'][i]) is str:
            nameint = (float(len(df['Path'][i])))
        else:
            nameint = (float(df['Path'][i]))
        df['Path'][i] = nameint
        if type(df['City'][i]) is str:
            nameint = (float(len(df['City'][i])))
        else:
            nameint = (float(df['City'][i]))
        df['City'][i] = nameint
        df['EntryHeading'][i] = headings.get(df['EntryHeading'][i])
        df['ExitHeading'][i] = headings.get(df['ExitHeading'][i])

def csv_to_tensor(trainfile):
    n = sum(1 for line in open(trainfile)) - 1 #number of records in file (excludes header)
    s = 10000 #desired sample size
    skip1 = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
    skip2 = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
    df = pd.read_csv(trainfile, skiprows=skip1)
    dft = pd.read_csv(trainfile, skiprows=skip2)
    floatify_df(df, s)
    floatify_df(dft, s)
    df = df.apply(pd.to_numeric, errors='raise')
    dft = dft.apply(pd.to_numeric, errors='raise')
    input_cols = ['RowId','IntersectionId','Latitude','Longitude','EntryStreetName','ExitStreetName','EntryHeading','ExitHeading','Hour','Weekend','Month','Path','TotalTimeStopped_p40','TotalTimeStopped_p60','TimeFromFirstStop_p20','TimeFromFirstStop_p40','TimeFromFirstStop_p50','TimeFromFirstStop_p60','TimeFromFirstStop_p80','DistanceToFirstStop_p40','DistanceToFirstStop_p60','City']
    output_cols = ['TotalTimeStopped_p20','TotalTimeStopped_p50','TotalTimeStopped_p80','DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']
    inputs = torch.tensor(df[input_cols].values, dtype=torch.float32)
    targets = torch.tensor(df[output_cols].values, dtype=torch.float32)
    test_X = torch.tensor(dft[input_cols].values, dtype=torch.float32)
    test_Y = torch.tensor(dft[output_cols].values, dtype=torch.float32)
    # print(inputs.shape)
    # print(targets.shape)
    for col in inputs:
        if True in torch.isnan(col):
            col[4] = 16.0
            col[5] = 16.0
    for col in test_X:
        if True in torch.isnan(col):
            col[4] = 16.0
            col[5] = 16.0
    train = torch.utils.data.TensorDataset(inputs, targets)
    test = torch.utils.data.TensorDataset(test_X, test_Y)
    return train, test


# train, test = torch_digits()

# trains, tests = csv_to_tensor('train.csv')