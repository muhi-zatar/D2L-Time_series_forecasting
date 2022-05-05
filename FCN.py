import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(21, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

def prepare_data(train_file, val_file, test_file):
    train_data = pd.read_csv(train_file, index_col=0)
    val_data = pd.read_csv(val_file, index_col=0)
    test_data = pd.read_csv(test_file, index_col=0)

    return train_data, val_data, test_data


if __name__ == '__main__':


    #import pdb;pdb.set_trace()

    batch_size = 1

    train_data, val_data, test_data = prepare_data("train_data.csv", "val_data.csv", "test_data.csv")


#import pdb;pdb.set_trace()

    train_data, val_data, test_data = prepare_data("train_data.csv", "val_data.csv", "test_data.csv")
    train_norm = (train_data - train_data.mean()) / (train_data.max() - train_data.min())
    val_norm = (val_data - val_data.mean()) / (val_data.max() - val_data.min())
    test_norm =  (test_data - test_data.mean()) / (test_data.max() - test_data.min())

    features = [col for col in train_norm.columns if col != 'SI [MW]']


    X_train = train_data[features]
    y_train = train_data['SI [MW]']
    X_val = val_data[features]
    y_val = val_data['SI [MW]']
    X_test = test_data[features]
    y_test = test_data['SI [MW]']
#import pdb;pdb.set_trace()
    ss = StandardScaler()
    X_train_sc = ss.fit_transform(X_train)
    X_val_sc = ss.transform(X_val)
    X_test_sc = ss.transform(X_test)

    X1 = train_norm.drop('SI [MW]',axis=1)
    y1 = train_norm['SI [MW]']
    X2 = val_norm.drop('SI [MW]',axis=1)
    y2 = val_norm['SI [MW]']
    X3 = test_norm.drop('SI [MW]',axis=1)
    y3 = test_norm['SI [MW]']
    Xtrain = np.array(X1)
    ytrain  = np.array(y1)
    Xval = np.array(X2)
    yval = np.array(y2)
    Xtest = np.array(X3)
    ytest = np.array(y3)
    Xtrain = torch.Tensor(Xtrain)
    ytrain  = torch.Tensor(ytrain)
    Xval = torch.Tensor(Xval)
    yval = torch.Tensor(yval)
    Xtest = torch.Tensor(Xtest)
    ytest = torch.Tensor(ytest)

    train = TensorDataset(Xtrain, ytrain)
    val = TensorDataset(Xval, yval)
    test = TensorDataset(Xtest, ytest)
#import pdb;pdb.set_trace()
#test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
#test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    mlp = MLP()
  
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    for epoch in range(0, 5): # 5 epochs at maximum
    
    # Print epoch
        print(f'Starting epoch {epoch+1}')
    
    # Set current loss value
        current_loss = 0.0
    
    # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            #import pdb;pdb.set_trace()
      # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
      
      # Zero the gradients
            optimizer.zero_grad()
      
      # Perform forward pass
            outputs = mlp(inputs.reshape(1,-1))
      
      # Compute loss
            loss = loss_function(outputs, targets)
      
      # Perform backward pass
            loss.backward()
      
      # Perform optimization
            optimizer.step()
      
      # Print statistics
            current_loss += loss.item()
            if i % 1000 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0

  # Process is complete.
    print('Training process has finished.')
