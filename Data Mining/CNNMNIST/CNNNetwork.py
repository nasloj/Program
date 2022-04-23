#CNNNetwork.py
import torch
import sys
import torch.nn as nn
import torchvision
# CNN network followed by a with one hidden layer classifier
class CNNNetwork(nn.Module):
    def __init__(self, hidden_size1, num_classes):
        super(CNNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 6 feature maps, 5x5 conv.
        # 1 is the number of channels. This is because of gray scale image 
        self.pool = nn.MaxPool2d(2,2)
        
        # we use the maxpool multiple times, but define it once
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6,8,3)
        
        # 6 channels, 8 feature maps, and 3x3 convolution
        self.fc1 = nn.Linear(8*5*5, hidden_size1) 
        self.fc2 = nn.Linear(hidden_size1, num_classes) 
        self.smax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(self.relu(out)) 
        out = self.conv2(out)
        out = self.pool(self.relu(out)) 
        flatten = out.view(-1,8*5*5) 
        out = self.fc1(flatten)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.smax(out)
        return out