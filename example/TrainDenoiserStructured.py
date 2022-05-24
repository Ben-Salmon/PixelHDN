import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
sys.path.append('../')
from HDN.models.lvae import LadderVAE
from utils.dataloaders import create_dn_loader
from tifffile import imread
from noise_models.PixelCNN import PixelCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

observation = imread('../data/MRI/observation.tif')
noise = imread('../data/MRI/noise.tif')

transform = transforms.RandomCrop(128)
train_loader, val_loader, data_mean, data_std, img_shape = create_dn_loader(observation, batch_size=8, split=0.9)
del(observation)

gaussian_noise_std = np.std(noise)
del(noise)
    
lr=3e-4
num_latents = 5
z_dims = [32]*int(num_latents)
free_bits = 0.5
use_uncond_mode_at=[0,1,2]

model = LadderVAE(z_dims=z_dims,
                  data_mean=data_mean,
                  data_std=data_std,
                  gaussian_noise_std=gaussian_noise_std,
                  img_shape=img_shape,
                  img_folder='../dn_checkpoint/MRI/HDN/structured/imgs',
                  use_uncond_mode_at=use_uncond_mode_at)

checkpoint_path = '../dn_checkpoint/MRI/HDN/structured'
trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, "HDN"),
                     gpus=1 if str(device).startswith("cuda") else 0,
                     max_epochs=6000,
                     logger=False,
                     gradient_clip_val=0,
                     callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_elbo", every_n_epochs=50),
                                EarlyStopping('val_elbo', patience=6000)])

trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('../dn_checkpoint/MRI/HDN/structured/final_params.ckpt')