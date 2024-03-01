import numpy as np
import pandas as pd
import torch
import random
from torch import dropout, nn
from torch.utils import data

train = pd.read_csv('facebook_train.csv')
test = pd.read_csv('facebook_test.csv')

train.isnull().any(axis=1).sum()
train.isnull().any(axis=1).sum()

train.dropna(axis=0, inplace=True)

# train.dtypes.value_counts()
# train.shape
# train.head()

features, labels = torch.tensor(train.iloc[:, 1:].to_numpy()), torch.tensor(train.iloc[:, 0].to_numpy().reshape((-1,1)))

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size, lr, num_epochs = 256, 3e-4, 500

data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(
    nn.Linear(155, 256), nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(64, 1))

net = net.double()

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

net.apply(init_weights)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X.double()), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

t = test.dropna(axis=0)
test_ = torch.Tensor(t.iloc[:, 1:].to_numpy())

yhat = net(test_.double())

def mse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    diff = np.subtract(actual, predicted)
    sq_diff = np.square(diff)
    return sq_diff.mean()

import math
math.sqrt(mse(yhat.detach().numpy(), t.iloc[:, 0]))