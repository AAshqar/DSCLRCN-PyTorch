"""LocalFeaturesCNN"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_vgg16.local_cnn import LocalFeatsCNN
from models.places_vgg16.places_cnn import PlacesCNN


class DSCLRCN(nn.Module):

    def __init__(self, input_dim=(360, 480), LSTMs_input_size=(512*30, 512*15), LSTM_hs=256):
        super(DSCLRCN, self).__init__()
		
		self.input_dim = input_dim
		self.LSTMs_input_size = LSTMs_input_size
		
		self.local_feats = LocalFeatsCNN()
		
		self.context = PlacesCNN()
		
        self.fc_h = nn.Linear(128, LSTMs_input_size[0])
        self.fc_v = nn.Linear(128, LSTMs_input_size[1])
        
		self.lstm_h = nn.LSTM(LSTMs_input_size[0], LSTM_hs, 1, batch_first=True)
		self.lstm_v = nn.LSTM(LSTMs_input_size[1], LSTM_hs, 1, batch_first=True)
		
		self.last_conv = nn.Conv2d(512, 1, 1)
		
		self.score = nn.Softmax()
		
		self.upsample = nn.Upsample(size=input_dim)
		

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        N = x.size(0)
        H,W = self.input_dim
        
        local_feats = self.local_feats(x)
        H_lf, W_lf = local_feats.size()[2:]
        context = self.context(x)
        
        perm_h = torch.LongTensor(np.arange(H-1, -1, -1))
        perm_v = torch.LongTensor(np.arange(W-1, -1, -1))
        
        context_h = self.fc_h(context)
        context_h = context_h.view(N, 1, LSTMs_input_size[0])
        local_feats_h = local_feats.view(N, H, LSTMs_input_size[0])
        local_feats_h1 = torch.cat((context_h, local_feats_h), dim=1)
        local_feats_h2 = local_feats_h[:, perm_h, :]
        local_feats_h2 = torch.cat((context_h, local_feats_h2), dim=1)
        
        output_h1, hz1 = self.lstm_h(local_feats_h1)
        output_h1 = output_h1[:,1:,]
        output_h1 = output_h1.view(N, 512, H_lf, W_lf)
        
        output_h2, hz2 = self.lstm_h(local_feats_h2)
        output_h2 = output_h2[:,1:,]
        output_h2 = output_h2.view(N, 512, H_lf, W_lf)
        
        output_h12 = torch.cat((output_h1, output_h2), dim=1)
        
        context_v = self.fc_v(context)
        context_v = context_v.view(N, 1, LSTMs_input_size[1])
        output_h12v = output_h12.view(N, W, LSTMs_input_size[1])
        output_h12v1 = torch.cat((context_h, output_h12v), dim=1)
        output_h12v2 = output_h12v[:, perm_v, :]
        output_h12v2 = torch.cat((context_v, output_h12v2), dim=1)
        
        output_h12v1, hz3 = self.lstm_v(output_h12v1)
        output_h12v1 = output_h12v1[:,1:,]
        output_h12v1 = output_h12v1.view(N, 512, H_lf, W_lf)
        
        output_h12v2, hz4 = self.lstm_v(output_h12v2)
        output_h12v2 = output_h12v2[:,1:,]
        output_h12v2 = output_h12v2.view(N, 512, H_lf, W_lf)
        
        output_h12v12 = torch.cat((output_h12v1, output_h12v2), dim=1)
        
        output_score = self.score(self.last_conv(output_h12v12))
        
        upsampled_score = self.upsample(output_score)

        return upsampled_score

    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
