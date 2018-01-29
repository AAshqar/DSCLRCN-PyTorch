"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation_resnet50.models import ModelBuilder


class SegmentationNN(nn.Module):

    def __init__(self, models_path='models/segmentation_resnet50/'):
        super(SegmentationNN, self).__init__()

        encoder_path = models_path+'encoder_best.pth'
        decoder_path = models_path+'decoder_best.pth'

        builder = ModelBuilder()
        self.net_encoder = builder.build_encoder(arch='resnet50_dilated8', fc_dim=512, weights=encoder_path)
        
        self.trans_feats = nn.Conv2d(2048, 512, kernel_size=(7, 5), stride=(3, 2), padding=(2, 2))

        #self.net_encoder.eval()
        #self.net_decoder = builder.build_decoder(arch='psp_bilinear', fc_dim=2048, weights=decoder_path)
        #self.net_decoder.eval()
        

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        x = self.net_encoder(x)
        #print(x.size())
        #x = self.net_decoder(x, segSize=(x.size(-2), x.size(-1)))
        
        x = self.trans_feats(x)
        
        #x = self.net_decoder(self.net_encoder(x), segSize=(x.size(-2), x.size(-1)))

        return x

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
