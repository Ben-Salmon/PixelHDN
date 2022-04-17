import os
import torch
import numpy as np

from tifffile import imread, imsave

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/rds/projects/k/krullaff-ibin/ben/PixelHDN16-04/PixelHDN16-04')
from noise_models.ShiftNet import ShiftNet

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import torch.optim as optim
from torchvision import transforms

from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import torch.utils.data as dt


simdata = (np.random.randn(2000,256,256)).astype(np.float32)
simdata = gaussian_filter(simdata, (0,0,4))
simdata = simdata + np.random.randn(2000,256,256)*0.1
simdata = simdata-simdata.mean()
simdata = simdata[:,np.newaxis,...]*150.0
simdata = simdata - simdata.mean()
dataStd = np.std(simdata)

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
dataset = MySimDataset(simdata, transform=transform)

# Split dataset into training and validation subsets
train_set, val_set = torch.utils.data.random_split(dataset, [1800, 200])

# Create pytorch dataloader for training and validation sets
train_loader = dt.DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True, pin_memory=True)
val_loader = dt.DataLoader(val_set, batch_size=16, shuffle=False, drop_last=False)

CHECKPOINT_PATH = 'nm_checkpoint'
trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ShiftNet"),
                     gpus=1 if str(device).startswith("cuda") else 0,
                     max_epochs=2000,
                     callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_nll"),
                                LearningRateMonitor("epoch"),
                                EarlyStopping('val_nll', patience=20)])

model = ShiftNet(kernel_size = 7,
                 depth =5, num_filters = 128,
                 num_gaussians = 10,
                 std = dataStd).to(device)

trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('nm_checkpoint/noiseModel.ckpt')
