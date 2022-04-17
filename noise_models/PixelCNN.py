import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from divnoising.AutoRegressiveGMM import AutoRegressiveGMM
import torch.optim as optim

'''
Code by Hrituraj Singh
Indian Institute of Technology Roorkee
'''

class MaskedConv2d(nn.Conv2d):
	"""
	Implementation of Masked CNN Class as explained in A Oord et. al. 
	Taken from https://github.com/jzbontar/pixelcnn-pytorch
	"""

	def __init__(self, mask_type, *args, **kwargs):
		self.mask_type = mask_type
		assert mask_type in ['A', 'B'], "Unknown Mask Type"
		super(MaskedConv2d, self).__init__(*args, **kwargs)
		self.register_buffer('mask', self.weight.data.clone())

		_, depth, height, width = self.weight.size()
    # Create a mask for nn.Conv2d convolutions
		self.mask.fill_(1)
		if mask_type =='A':
			self.mask[:,:,height//2,width//2:] = 0
			self.mask[:,:,height//2+1:,:] = 0
		else:
			self.mask[:,:,height//2,width//2+1:] = 0
			self.mask[:,:,height//2+1:,:] = 0


	def forward(self, x):
     # Apply mask to convolutions
		self.weight.data*=self.mask
		return super(MaskedConv2d, self).forward(x)

class PixelCNN(AutoRegressiveGMM):
    def __init__(self, in_channels=1,
                 hidden_channels=64,
                 kernel_size=(5,5),
                 depth=12,
                 num_gaussians=10,
                 mean=0, std=1):
        super().__init__(mean,std,num_gaussians)
        self.save_hyperparameters()
        num_params = num_gaussians*3
        # Define layers
        self.convs = [MaskedConv2d('A', in_channels, hidden_channels, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2))]
        self.convs.append(nn.ReLU())
        for i in range(depth):
            self.convs.append(MaskedConv2d('B', hidden_channels, hidden_channels, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)))
            self.convs.append(nn.ReLU())
        self.convs.append(MaskedConv2d('B', hidden_channels, in_channels*num_params, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)))

        self.convs = nn.ModuleList(self.convs)

        self.example_input_array = train_set[0][None]

        self.reset_params()

    @staticmethod
    def weight_init(m):
        # Initialise weights
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        # Computes one forward pass
        for i, module in enumerate(self.convs):
            x = module(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
