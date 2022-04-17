import os
import torch
import numpy as np

from tifffile import imread, imsave

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/rds/projects/k/krullaff-ibin/ben/PixelHDN16-04/PixelHDN16-04')
from tifffile import imread
from noise_models.ShiftNet import ShiftNet
from denoisers.HDN.models.lvae import LadderVAE

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import time
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.optim as optim
from torchvision import transforms

from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import torch.utils.data as dt


simdata = (np.random.randn(2450,256,256)).astype(np.float32)
simdata = gaussian_filter(simdata, (0,0,4))
simdata = simdata + np.random.randn(2450,256,256)*0.1
simdata = simdata-simdata.mean()
simdata = simdata[:,np.newaxis,...]*150.0
simdata = simdata - simdata.mean()
dataStd = np.std(simdata)

signal = imread('data/signal.tif')
observation = signal[:2450] + simdata

class MySimDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.from_numpy(data).type(torch.float)
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        img = self.data[idx]
        
        if self.transform:
            img = transform(img)
            
        return img

# Choose transformations
transform = transforms.RandomCrop(64)
# Create pytorch dataset
dataset = MySimDataset(observation, transform=transform)

# Split dataset into training and validation subsets
train_set, val_set = torch.utils.data.random_split(dataset, [2200, 250])

# Create pytorch dataloader for training and validation sets
train_loader = dt.DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True, pin_memory=True)
val_loader = dt.DataLoader(val_set, batch_size=16, shuffle=False, drop_last=False)

data_mean = np.mean(observation)
data_std = np.std(observation)

noise_model = ShiftNet.load_from_checkpoint('nm_checkpoint/noiseModel.ckpt').to(device)

CHECKPOINT_PATH = 'dn_checkpoint'
trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "HDN"),
                     gpus=1 if str(device).startswith("cuda") else 0,
                     max_epochs=2000,
                     logger=False,
                     callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_elbo"),
                                EarlyStopping('val_elbo', patience=20)],
                     gradient_clip_val=0.5)

img_shape = dataset[0].shape[-2:]
num_latents = 5
z_dims = [32]*int(num_latents)
HDN = LadderVAE(z_dims=z_dims, data_mean=data_mean, data_std=data_std, 
                virtual_batch=8, lr=3e-4, weight_decay=0, device=device,
                noiseModel=noise_model, free_bits=1.0, 
                img_shape=img_shape).to(device)

trainer.fit(HDN, train_loader, val_loader)
trainer.save_checkpoint('dn_checkpoint/HDN.ckpt')