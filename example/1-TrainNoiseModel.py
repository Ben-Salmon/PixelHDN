'''
import warnings
warnings.filterwarnings('ignore')
import torch
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
from tifffile import imread
import sys
sys.path.append('../')
#from pn2v import *
from denoisers.HDN.lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel

observation = imread('../data/calibration_data.tif')
nameGMMNoiseModel = 'GMMNoiseModel'
path = '../nm_checkpoint/'
n_gaussian = 3 # Number of gaussians to use for Gaussian Mixture Model
n_coeff = 2 # No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components.

signal=np.mean(observation[:, ...],axis=0)[np.newaxis,...]

min_signal=np.min(signal)
max_signal=np.max(signal)

gaussianMixtureNoiseModel = GaussianMixtureNoiseModel(min_signal = min_signal,
                                                      max_signal =max_signal,
                                                      path=path, weight = None, 
                                                      n_gaussian = n_gaussian,
                                                      n_coeff = n_coeff,
                                                      min_sigma = 50, 
                                                      device = device)

gaussianMixtureNoiseModel.train(signal, observation, batchSize = 250000, n_epochs = 2000, learning_rate=0.1, name = nameGMMNoiseModel)

'''
import os
import torch
import torch.utils.data as dt
import numpy as np
from tifffile import imread
import sys
sys.path.append('../')
from noise_models.ShiftNet import ShiftNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torchvision import transforms

# Load noise
noise = imread('../data/noise.tif')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, n_data, transform=None):
        self.n_data = torch.from_numpy(n_data).type(torch.float)
        
        self.transform = transform
        
        if self.n_data.dim() == 3:
            self.n_data = self.n_data[:,np.newaxis,...]
        elif self.n_data.dim() != 4:
            print('Data dimensions should be [B,C,H,W] or [B,H,W]')
    
    def getparams(self):
        return torch.mean(self.n_data), torch.std(self.n_data)
    
    def __len__(self):
        return self.n_data.shape[0]
    
    def __getitem__(self, idx):
        n = self.n_data[idx]
        
        if self.transform:
            n = self.transform(n)
        
        return n

# Choose transformations
transform = transforms.RandomCrop(64)
# Create pytorch dataset
dataset = MyDataset(noise, transform=transform)

# Split dataset into training and validation subsets
train_set, val_set = torch.utils.data.random_split(dataset, [round(len(dataset)*0.8), round(len(dataset)*0.2)])

# Create pytorch dataloader for training and validation sets
train_loader = dt.DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
val_loader = dt.DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False)

n_mean, n_std = dataset.getparams()

checkpoint_path = '../nm_checkpoint'
trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, "ShiftNet"),
                     gpus=1 if str(device).startswith("cuda") else 0,
                     max_epochs=2000,
                     callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_nll"),
                                LearningRateMonitor("epoch"),
                                EarlyStopping('val_nll', patience=10)])

model = ShiftNet(kernel_size = 7,
                 depth =5, num_filters = 128,
                 num_gaussians = 10,
                 mean = 0,
                 std = n_std).to(device)

trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('../nm_checkpoint/final_params.ckpt')