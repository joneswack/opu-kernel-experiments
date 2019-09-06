import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim, bias=False)
        torch.nn.init.zeros_(self.layer.weight)
        
    def forward(self, input):
        return self.layer.forward(input)
    
class BasicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class MultipleRegressionSolver(object):
    def __init__(self, X, Y, batch_size=1000, cuda=True):
        super(MultipleRegressionSolver, self).__init__()
        
        self.X = X
        # We assume Y to be of shape (n_samples, output_dim)
        self.Y = Y
        
        self.dataset = BasicDataset(self.X, self.Y)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.model = RegressionModel(self.X.shape[1], self.Y.shape[1])
        
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