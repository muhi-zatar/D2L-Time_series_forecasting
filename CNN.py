import numpy as np # linear algebra
import pandas as pd 

from numpy import array
import torch
import gc
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader, TensorDataset


class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()
        self.conv1d = nn.Conv1d(5,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64*64,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        #import pdb;pdb.set_trace()
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

#def prepare_data(load_file, pv_file, wind_off_file, wind_on_file, power_file, prices_file):
#    load = pd.read_excel(load_file)
#    pv = pd.read_excel(pv_file)
#    wind_off = pd.read_excel(wind_off_file)
#    wind_on = pd.read_excel(wind_on_file)
#    power = pd.read_excel(power_file)
#    prices = pd.read_excel(prices_file)

    #import pdb;pdb.set_trace()
#    return load["Total_Load_measured"].to_numpy(), pv["RT_pv_MW"].to_numpy(), wind_off["RT_wind_off_shore_MW"].to_numpy(), wind_on["RT_wind_on_shore_MW"].to_numpy(), power["Total"].to_numpy(), prices["Day-ahead Price [€\MWh]"].to_numpy()

def prepare_data(train_file, val_file, test_file):
    train_data = pd.read_csv(train_file, index_col=0)
    val_data = pd.read_csv(val_file, index_col=0)
    test_data = pd.read_csv(test_file, index_col=0)

    return train_data, val_data, test_data

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

batch_size = 64

train_data, val_data, test_data = prepare_data("train_data.csv", "val_data.csv", "test_data.csv")

load_t = torch.Tensor(load)
pv_t = torch.Tensor(pv)
wind_off_t = torch.Tensor(wind_off)
wind_on_t = torch.Tensor(wind_on)
power_t = torch.Tensor(power)
train_targets = torch.Tensor(prices[:145920].reshape(-1,1))

#import pdb;pdb.set_trace()

train_features = torch.stack((load_t[:145920].reshape(-1,1), pv_t[:145920].reshape(-1,1), wind_on_t[:145920].reshape(-1,1), power_t[:145920].reshape(-1,1), wind_off_t[:145920].reshape(-1,1)), dim=1)
val_features = train_features[140000:]
val_targets = train_targets[140000:]
train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(val_features, val_targets)
#import pdb;pdb.set_trace()
#test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
#test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN_ForecastNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

train_losses = []
valid_losses = []

def Train():
    
    running_loss = .0
    
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    #train_losses.append(train_loss.detach().numpy())
    
    print(f'train_loss {train_loss}')
    
def Valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(val_loader)
        #valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss {valid_loss}')

epochs = 200
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Valid()
    gc.collect()
