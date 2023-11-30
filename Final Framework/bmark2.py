from time import time
import bmark2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import random
import time



class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!
        
        self.conv18 = nn.Conv2d(1, 8, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size = 2)
        self.conv84 = nn.Conv2d(8, 4, kernel_size=3)
        self.linear = nn.Linear(22, 8)
        self.linearx2 = nn.Linear(10, 30)
        self.bilinear = nn.Bilinear(44, 10, 24)
        self.output = nn.Linear(8, 6)
        #self.relu = F.relu
        pass

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (1,8,8) torch tensor.

        Returns:
            An (N, 6) torch tensor
        '''
        #xb = xb.unsqueeze(1)
        #print(xb.size())
        N = xb.size()[0]
        #print('xb size: ', xb.shape)
        # y = F.relu(self.conv18(xb))
        # y = F.relu(self.maxpool(y))
        # y = F.relu(self.conv84(y))
        # y = y.view(N,4)
        #print(y.size())
        #y = self.norm(xb)
        y = F.relu(self.linear(xb))
        #y = F.relu(self.linearx2(y))
        #y = F.relu(self.bilinear(x, y))
        y = F.leaky_relu(self.output(y))
        return y
        #return(model_1(xb))
        pass

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    gamma = 1.0
    if(batch_size == 2):
        batch_size = 1
        gamma = 0.95
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)
    train_losses = []
    test_losses = []
    #loss = hw2_utils.epoch_loss(net, loss_func, train_dl)
    #print(hw2_utils.epoch_loss(net, loss_func, train_dl)) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # with torch.no_grad():
    #     train_losses.append(bmark2_utils.epoch_loss(net, loss_func, train_dl))
    #     test_losses.append(bmark2_utils.epoch_loss(net, loss_func, train_dl))
    for i in range(n_epochs): #training
        print('epoch: ', i, end=' ')
        for i, (X_batch, Y_batch) in enumerate(train_dl): 
            bmark2_utils.train_batch(net, loss_func, X_batch, Y_batch, optimizer)
        with torch.no_grad():
            trainloss = bmark2_utils.epoch_loss(net, loss_func, train_dl)
            print( 'loss: ', trainloss)
            train_losses.append(trainloss)
            test_losses.append(bmark2_utils.epoch_loss(net, loss_func, test_dl))
        scheduler.step()
    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    return train_losses, test_losses

train, test = bmark2_utils.csv_to_tensor('train.csv')
net1 = DigitsConvNet()
optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.1)
# net2 = DigitsConvNet()
# optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.005)
# net3 = DigitsConvNet()
# optimizer3 = torch.optim.SGD(net3.parameters(), lr=0.005)
start_time = time.time()
train1, test1 = fit_and_evaluate(net1, optimizer1, torch.nn.MSELoss(), train, test, 1000, 10)
#print(train1)
# train2, test2 = fit_and_evaluate(net2, optimizer2, torch.nn.CrossEntropyLoss(), train, test, 30, 2)
# train3, test3 = fit_and_evaluate(net3, optimizer3, torch.nn.CrossEntropyLoss(), train, test, 30, 16)
# #print(train1)
plt.plot(range(len(train1)), train1, c='#ff0000')
# plt.plot(range(len(train2)), train2, c='#00ff00', marker='.')
# plt.plot(range(len(train3)), train3, c='#0000ff', marker='.')
plt.show()
plt.clf()
plt.plot(range(len(test1)), test1, c='#00ff00')
# plt.plot(range(len(test2)), test2, c='#88ff88', marker='.')
# plt.plot(range(len(test3)), test3, c='#8888ff', marker='.')
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
# print('Training Accuracy: %.2f' % bmark2_utils.compute_accuracy(net1, train))
# print('Test Accuracy: %.2f' % bmark2_utils.compute_accuracy(net1, test))
plt.show()


