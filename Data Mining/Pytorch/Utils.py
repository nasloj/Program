#Utils.py
import torch
from MyDataSet import MyDataSet
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split 
import numpy as np

def get_train_val_loaders(batch_size):
    x_train = np.arange(0,10)
    y_train=x_train*2+0.3 #y=2x+0.3
    x_train_tensor = torch.from_numpy(x_train).float() 
    y_train_tensor = torch.from_numpy(y_train).float()
    mydataset = MyDataSet(x_train_tensor, y_train_tensor) 
    train_dataset, val_dataset = random_split(mydataset, [8, 2])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size) 
    return train_loader, val_loader
