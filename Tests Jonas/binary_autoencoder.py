import torch
import torch.nn as nn
from torch import optim

import numpy as np

class BinaryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, beta=1.):
        """Autoencoder.
        Parameters
        ----------
        input_size: int,
            input size of the model
        hidden_size: int,
            size of the hidden layer
        beta: float,
            coefficient for tanh(beta * x)
        """
        super(BinaryEncoder, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.beta = beta
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.encoder.weight, gain = 5/3)
        nn.init.xavier_uniform_(self.decoder.weight, gain = 5/3)
    
    def forward(self, images):
        """Computes binary code with annealing on tanh(beta * x) and reconstructs the image."""
        x = images#.view(-1,self.input_size)  # it flattens a 2d image into a vector
        h = self.tanh(self.beta * self.encoder(x)) / self.beta  # binary code
        r = self.decoder(h) # reconstruction
        return r 
      
    def encoder_isolate(self, images):
        """Returns binary code using torch.sign"""
        x = images.view(-1, self.input_size)
        h = (torch.sign(self.encoder(x)) + 1) / 2  # self.tanh(self.beta * self.encoder(x))
        return h
    
    def train(self, n_epochs, dataloader, lr = 1e-4):
        """Training loop. Returns the trained model."""
        criterion = nn.MSELoss()
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=lr)
        # loop on epochs
        for e in range(n_epochs):
            # gradually increasing beta by factor 10 at every 10 epochs
            if e != 0:
                if e % 2 == 0:
                    self.beta = self.beta * 2
            # inner training loop

            losses = []
            for i, (data) in enumerate(dataloader):
                optimizer.zero_grad()
                images = data[0].cuda()
                output = self.forward(images)
                loss = criterion(output, images)#.view(-1, 18)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                
                if i % 100:
                    print('Batch:', i, 'Loss:', loss.item())

            print('Epoch:', e, 'Loss:', np.mean(losses))