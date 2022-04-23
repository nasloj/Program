#Utils.py
import torch
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt

def get_loaders(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(),download=True) 
    test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True) 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
def plot_images(loader):
    mnistdata = iter(loader)
    digit, label = mnistdata.next() 
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(digit[i][0], cmap='gray') 
    plt.show()