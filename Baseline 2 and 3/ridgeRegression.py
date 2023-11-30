from tabnanny import verbose
from turtle import shape
import bmark2_utils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import io
import time
from torch.utils.data import TensorDataset, DataLoader
import random
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

class DisplayLossCurve(object):
  def __init__(self, print_loss=False):
    self.print_loss = print_loss

  """Make sure the model verbose is set to 1"""
  def __enter__(self):
    self.old_stdout = sys.stdout
    sys.stdout = self.mystdout = io.StringIO()
  
  def __exit__(self, *args, **kwargs):
    sys.stdout = self.old_stdout
    loss_history = self.mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
      if(len(line.split("loss: ")) == 1):
        continue
      loss_list.append(float(line.split("loss: ")[-1]))
      #print(line)
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if self.print_loss:
      print("=============== Loss Array ===============")
      print(np.array(loss_list))
      
    return True

def csv_to_tensor(trainfile):
    n = sum(1 for line in open(trainfile)) - 1 #number of records in file (excludes header)
    s = 5000 #desired sample size
    skip1 = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
    skip2 = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
    df = pd.read_csv(trainfile, skiprows=skip1)
    dft = pd.read_csv(trainfile, skiprows=skip2)
    bmark2_utils.floatify_df(df, s)
    bmark2_utils.floatify_df(dft, s)
    df = df.apply(pd.to_numeric, errors='raise')
    dft = dft.apply(pd.to_numeric, errors='raise')
    input_cols = ['RowId','IntersectionId','Latitude','Longitude','EntryStreetName','ExitStreetName','EntryHeading','ExitHeading','Hour','Weekend','Month','Path','TotalTimeStopped_p40','TotalTimeStopped_p60','TimeFromFirstStop_p20','TimeFromFirstStop_p40','TimeFromFirstStop_p50','TimeFromFirstStop_p60','TimeFromFirstStop_p80','DistanceToFirstStop_p40','DistanceToFirstStop_p60','City']
    output_cols = ['TotalTimeStopped_p20','TotalTimeStopped_p50','TotalTimeStopped_p80','DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']
    inputs = torch.tensor(df[input_cols].values, dtype=torch.float32)
    targets = torch.tensor(df[output_cols].values, dtype=torch.float32)
    test_X = torch.tensor(dft[input_cols].values, dtype=torch.float32)
    test_Y = torch.tensor(dft[output_cols].values, dtype=torch.float32)
    #print(inputs.shape)
    for col in inputs:
        if True in torch.isnan(col):
            col[4] = 16.0
            col[5] = 16.0
    for col in test_X:
        if True in torch.isnan(col):
            col[4] = 16.0
            col[5] = 16.0
    #print(targets.shape)
    return inputs, targets, test_X, test_Y

inputs, targets, test_X, test_Y = csv_to_tensor('train.csv')


reg = Ridge()
reg.fit(inputs, targets)

start_time = time.time()
train_sizes, train_scores, validation_scores = learning_curve(
estimator = Ridge(),
X = inputs, y = targets, 
cv = 5, scoring = 'neg_mean_squared_error')

print(train_scores.shape)
print(validation_scores.shape)

print('Training scores:\n\n', train_scores)

print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = -train_scores.mean(axis = 1)

print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator


plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error', c='#ff0000')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()
print('Total Training Time: %.2f sec' % ((time.time() - start_time)))
plt.show()
print('Training Score:', reg.score(inputs, targets))
print('Testing Score:', reg.score(test_X, test_Y))
# regr = LogisticRegression(**params)
# with DisplayLossCurve(print_loss=True):
#     regr.fit(inputs, targets)

# pred = regr.predict(test_X)

# The coefficients
# print("Coefficients: \n", regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(test_Y, pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(test_Y, pred))



# plt.show()


