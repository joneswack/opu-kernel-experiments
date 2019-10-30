import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import glob
from time import time
from datetime import datetime

from tensorboardX import SummaryWriter

def clean_dir(path):
    files = glob.glob(path + '/*')
    for f in files:
        os.remove(f)

class ModelTrainer():
    
    def __init__(self, model_name, model, train_loader, test_loader,
            lr=1e-3, epochs=30, use_gpu=False):
        super(ModelTrainer, self).__init__()
        
        if use_gpu:
            model.cuda()
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs
        self.use_gpu = use_gpu

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        
        # we should have one log dir per run
        # otherwise tensorboard will have overlapping graphs
        self.log_dir = 'tensorboard_logs/{}_lr_{}_epochs_{}/{}'.format(
            model_name, lr, epochs, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = SummaryWriter(self.log_dir + '/train')
        self.test_writer = SummaryWriter(self.log_dir + '/test')
        
    def run(self):
        t0 = time()
        
        # first test loss before training
        old_test_acc = self.test_epoch(self.model, self.test_loader, 0)
        
        for epoch in range(1, self.epochs + 1):
            print('\n# Epoch {} #\n'.format(epoch))
            self.scheduler.step(epoch)
            self.train_epoch(self.model, self.train_loader, self.optimizer, epoch)
            test_acc = self.test_epoch(self.model, self.test_loader, epoch)
            
            # if test_acc > old_test_acc:
            self.save(self.model)
            # old_test_acc = test_acc

        time_elapsed = time() - t0
        print('\nTime elapsed: {:.2f} seconds'.format(time_elapsed))
        self.train_writer.close()
        self.test_writer.close()
        
        return time_elapsed

    def train_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        train_loss = 0
        correct = 0
        runs = 1
        num_batches = len(train_loader.dataset)

        for i in range(runs):

            for batch_idx, (data, target) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # loss before updating the weights (i.e. at the beginning of each iteration)
                iteration = (epoch-1) * len(train_loader) + batch_idx

                if iteration % 50 == 0:
                    for name, param in model.named_parameters():
                        self.train_writer.add_scalar(name + '_grad_norm', param.grad.norm(), iteration)
                        self.train_writer.add_histogram(name + '_grad', param.grad.clone().cpu().numpy(), iteration)
                        self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration)
                    
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                self.train_writer.add_scalar('nll_loss', loss / len(data), iteration)
            
        train_loss /= num_batches
        train_acc = 100. * correct / num_batches
        
        self.train_writer.add_scalar('accuracy', train_acc, iteration)

        print('[Train] Avg. Loss: {:.2f}, Avg. Accuracy: {:.2f}%'.format(
            train_loss, train_acc))

    def test_epoch(self, model, test_loader, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        
        # iteration after processing all batches of the current epoch
        iteration = epoch * len(self.train_loader)
        self.test_writer.add_scalar('nll_loss', test_loss, iteration)
        self.test_writer.add_scalar('accuracy', test_acc, iteration)

        print('[Test] Average loss: {:.2f}, Accuracy: {:.2f}%'.format(
            test_loss, test_acc))
        
        return test_acc
        
    def save(self, model):
        model_out_path = "models/state_dicts/alexnet_manual_train.pth"
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
