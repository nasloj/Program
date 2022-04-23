#CancerClassifications.py
import sys
import Utils
from Network import Network
import torch
import torch.nn as nn

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size1 = 100
    num_classes = 5
    num_epochs = 10
    batch_size = 10
    learning_rate = 0.001
    train_loader, test_loader = Utils.get_train_test_loaders(batch_size)

    
    model = Network(hidden_size1, num_classes).to(device)
    criterion = nn.CrossEntropyLoss() # for multiclass
    # classification, cross entropy loss works better
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    num_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (x, labels) in enumerate(train_loader):
            x = x.reshape(-1,1,x.size(1)).to(device) # convert to CPU or GPU
            
            labels = labels.type(torch.LongTensor) 
            labels = labels.to(device)
            
            pred_outputs = model(x) # calls forward function
            loss = criterion(pred_outputs, labels) 
            optimizer.zero_grad() # clear gradients 
            loss.backward() # compute gradients 
            optimizer.step() # update weights and biases 
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{num_total_steps}], Loss: {loss.item():.4f}')
                # compute accuracy on test set
            with torch.no_grad():
                num_correct = 0
                num_samples = 0
                for xt, labels in test_loader:
                    xt = xt.reshape(-1,1,xt.size(1)).to(device)
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(device)
                    outputs = model(xt)
                    _, predicted = torch.max(outputs, 1) # returns max,max_indices num_samples += labels.size(0)
                    num_correct += (predicted == labels).sum()
            acc = 100.0 * num_correct / num_samples
            print(f'Accuracy of the network on the test set: {acc} %')
            
if __name__ == "__main__":
    sys.exit(int(main() or 0))