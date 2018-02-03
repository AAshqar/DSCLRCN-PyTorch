from random import shuffle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from random import *


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.KLDivLoss()):
        
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        #if torch.cuda.is_available():
        #    model.cuda()

        print('START TRAIN.')
        
        nIterations = num_epochs*iter_per_epoch
        
        for j in range(num_epochs):
            for i, data in enumerate(train_loader, 0):
                
                it = j*iter_per_epoch + i
                inputs, labels = data
                #if torch.cuda.is_available():
                #    inputs, labels = inputs.cuda(), labels.cuda()
                
                inputs = Variable(inputs)
                labels = Variable(labels)
                model.train()
                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                if it%log_nth==0:
                    print('[Iteration %i/%i] TRAIN loss: %f' % (it, nIterations, loss))
                    self.train_loss_history.append(loss.data[0])
            
            model.eval()
            
            rand_select = randint(0, len(val_loader)-1)
            for ii, data in enumerate(val_loader, 0):
                if rand_select == ii:
                    inputs_val = Variable(torch.from_numpy(val_loader.dataset.X))
                    labels_val = Variable(torch.from_numpy(val_loader.dataset.y))
                    outputs_val = model.forward(inputs_val)
                    val_loss = self.loss_func(outputs_val, labels_val)
                    self.val_loss_history.append(loss.data[0])
            print('[Epoch %i/%i] TRAIN KLD Loss: %f' % (j, num_epochs, loss))
            print('[Epoch %i/%i] VAL KLD Loss: %f' % (j, num_epochs, val_loss))
            
        
        print('FINISH.')
