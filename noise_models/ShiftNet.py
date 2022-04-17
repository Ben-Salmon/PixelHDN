import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from noise_models.AutoRegressiveGMM import AutoRegressiveGMM
import torch.optim as optim
#import pytorch_lightning as pl

class ShiftNet(AutoRegressiveGMM):
    """ A conv-net that shifts it's output to modify the receptive field of ouput pixels

    """

    def __init__(self, 
                 in_channels=1, 
                 depth=5,
                 num_filters=128,
                 kernel_size=7,
                 num_gaussians=10,
                 mean=0, std=1):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        self.save_hyperparameters()
        super().__init__(mean, std, num_gaussians)
        
        
        
        self.num_gaussians = num_gaussians
        self.out_channels = num_gaussians * 3
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.depth = depth
        self.kernel_size = kernel_size       
        
        padding = (0,kernel_size//2)
        
        # this where we add our convolutions
        self.convs = []
        
        #initial convolution
        self.convs.append(nn.Conv2d(in_channels,
                                    num_filters,
                                    kernel_size = (1,kernel_size),
                                    padding = padding ))
                
        self.rf= kernel_size
               
        for i in range(self.depth-2):
            self.convs.append(nn.Conv2d(num_filters,
                            num_filters,
                            kernel_size = (1,kernel_size),
                            padding = padding))
            self.rf += (kernel_size-1)
                
        self.convs.append(nn.Conv2d(num_filters,
                            self.out_channels,
                            kernel_size = (1,1)))
        
                              
        # add the list of modules to current module
        self.convs = nn.ModuleList(self.convs)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)                     
            
    def forward(self, x):
        rf = self.rf
        rfh = rf//2 +1
        patchSize = x.shape[-1]
        x = torch.nn.functional.pad(x, (rf,rf,rf,rf),value = 5)


        for i, module in enumerate(self.convs[:-1]):
            x = module(x)
            if i < (self.depth-2):
                x = F.relu(x)
  
        medium = rf
        low = medium - rfh
        high = medium + rfh
        
        # These are differently shifted versions of the output
        # We can concatenate them to produce different receptive fields
        wa = x[...,low:low+patchSize, medium:medium+patchSize]
        wb = x[...,high:high+patchSize, medium:medium+patchSize]
        wc = x[...,medium:medium+patchSize, low:low+patchSize]
        wd = x[...,medium:medium+patchSize, high:high+patchSize]
        
        wll = x[...,low:low+patchSize, low:low+patchSize]
        wlh = x[...,low:low+patchSize, high:high+patchSize]
        whl = x[...,high:high+patchSize, low:low+patchSize]
        whh = x[...,high:high+patchSize, high:high+patchSize]
        
        wu = x[...,medium-rfh:medium-rfh+patchSize, medium:medium+patchSize]
        

        # Here we concatenate the different versions of the output
#        x = torch.cat((wc,wc),1)
        x = wc
        x = self.convs[-1](x)

        return x


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]