#LinearNetwork.py
import torch
import sys
import torch.nn as nn
import torchvision

# fully connected neural network with two hidden layers
class LinearNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(LinearNetwork, self).__init__() 
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2) 
        self.l3 = nn.Linear(hidden_size2, num_classes) 
        self.smax = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.smax(out)
        return out