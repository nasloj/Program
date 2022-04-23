#Pytorch.py
import sys
import numpy as np
import torch
import random
from SimpleModel import SimpleModel
import Utils
def main():
    z = np.arange(100)
#----------- use gpu if available else cpu------------ 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    train_loader, val_loader = Utils.get_train_val_loaders(2)
    losses = []
    val_losses = []
    lr = 1e-2
    n_epochs = 100
    loss_fn = torch.nn.MSELoss(reduction='mean')
    model = SimpleModel().to(device)
    model.train() # set the model in train mode
# tell optimizer to optimize the model paramaters, specify learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device) # load data in GPU if available 
            y_batch = y_batch.to(device)
            aout = model(x_batch) # implicitly calls forward function in model 
            loss = loss_fn(y_batch, aout)
            loss.backward()  # compute gradients
            optimizer.step() # update weights, biases
            optimizer.zero_grad() # clear gradients

            losses.append(loss)
        # do validation on the learned model so far
        with torch.no_grad():  # turn off gradient calculation
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                model.eval() # set model to evaluation mode
                aout = model(x_val)
                val_loss = loss_fn(y_val, aout) 
                val_losses.append(val_loss.item())
                print('epoch' + str(epoch) + ' validation loss = ' + str(val_loss))
    print(model.state_dict()) # print final model parameters
if __name__ == "__main__":
    sys.exit(int(main() or 0))