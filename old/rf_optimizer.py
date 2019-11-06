import torch
import torch.nn as nn

import random_features import RBFModulePyTorch

class RidgeRegressor(nn.Module):
    def __init__(self, Y_train, dtype=torch.FloatTensor):
        self.log_alpha = nn.Parameter(torch.ones(1).type(dtype), requires_grad=True)
        self.Y = Y_train
    
    def forward(self, x):
        # x are the kernel features
        A = x.t().mm(x) + torch.exp(self.log_alpha) * torch.eye(x.t().shape[0])
        b = x.t().mm(self.Y)
        
        L = torch.cholesky(A, upper=False, out=None)
        beta = torch.cholesky_solve(b, L)
        
        return x.mm(beta)
    
class RidgeRegressionSolver(object):
    def __init__(self, X, Y, cuda=True):
        super(RidgeRegressionSolver, self).__init__()
        
        self.X = X
        # We assume Y to be of shape (n_samples, output_dim)
        self.Y = Y
        
        self.dataset = BasicDataset(self.X, self.Y)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.model = RegressionModel(self.X.shape[1], self.Y.shape[1], feature_layer, zero_init)
        
        self.cuda = cuda
        
        if self.cuda:
            self.model = self.model.cuda()
        
    def fit(self, optimizer, epochs=10):
        criterion = torch.nn.MSELoss()
        
        for epoch in range(epochs):
            
            for step, (X, Y) in enumerate(self.data_loader):
                if self.cuda:
                    X = X.type(torch.FloatTensor).cuda()
                    Y = Y.type(torch.FloatTensor).cuda()
                    
                def closure():
                    optimizer.zero_grad()
                    output = self.model.forward(X)
                    loss = criterion(output, Y)
                    loss.backward()
                    
                    with torch.no_grad():
                        predictions = torch.argmax(output, 1)
                        batch_targets = torch.argmax(Y, 1)
                        correct = (predictions == batch_targets).sum().item()
                        total = Y.size(0)
                        
                        relative_error = ((output - Y).norm() / Y.norm()).item()
                        
                    if step == 0:
                        self.final_loss = loss.item()
                        print('Epoch:', epoch, 'Loss:', self.final_loss, 'Accuracy:', 100 * correct / total, 'Relative Error:', relative_error)

                    return loss
                
                optimizer.step(closure)
                
        # returns the weights of shape (in, out)
        return self.model.layer.weight.data.t().cpu().numpy(), self.final_loss
    
    def classification_score(self, data, targets):
        dataset = BasicDataset(data, targets)
        data_loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for step, (X, Y) in enumerate(self.data_loader):
                if self.cuda:
                    X = X.type(torch.FloatTensor).cuda()
                    Y = Y.type(torch.FloatTensor).cuda()

                output = self.model.forward(X)
                predictions = torch.argmax(output, 1)
                batch_targets = torch.argmax(Y, 1)

                correct += (predictions == batch_targets).sum().item()
                total += Y.size(0)
                
        accuracy = 100 * correct / total
        
        print('Accuracy:', accuracy)
        
        return accuracy