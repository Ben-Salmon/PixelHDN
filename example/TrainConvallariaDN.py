import warnings
warnings.filterwarnings('ignore')
import os
import torch
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
sys.path.append('../')
from HDN.models.lvae import LadderVAE
from utils.dataloaders import create_dn_loader
from tifffile import imread
from noise_models.PixelCNN_signaldependent import PixelCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

flower = imread('../data/flower.tif')

observation = flower[:,np.newaxis].astype(float)

transform = transforms.RandomCrop(64)
train_loader, val_loader, data_mean, data_std, img_shape = create_dn_loader(observation, transform, 0.8, 8)

noiseModel = PixelCNN.load_from_checkpoint('../nm_checkpoint/Convallaria/signal_dependent/final_params.ckpt').to(device).eval()
for param in noiseModel.parameters():
    param.requires_grad = False
    
lr=3e-4
num_latents = 7
z_dims = [32]*int(num_latents)
free_bits = 1

model = LadderVAE(z_dims=z_dims,
                   data_mean=data_mean,
                   data_std=data_std,
                   noiseModel=noiseModel,
                   img_shape=img_shape,
                   img_folder='../dn_checkpoint/Convallaria/signal_dependent/imgs').to(device)

checkpoint_path = '../dn_checkpoint/Convallaria/signal_dependent'
trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, "HDN"),
                     gpus=1 if str(device).startswith("cuda") else 0,
                     max_epochs=40000,
                     logger=False,
                     gradient_clip_val=0,
                     callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_elbo", every_n_epochs=50),
                                EarlyStopping('val_elbo', patience=40000)])

trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('../dn_checkpoint/Convallaria/signal_dependent/final_params.ckpt')