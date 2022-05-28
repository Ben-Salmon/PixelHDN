import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import sys
sys.path.append('../')
from noise_models.PixelCNN_signaldependent import PixelCNN
from utils.dataloaders_signaldependent import create_nm_loader
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flower = imread('../data/flower.tif')

observation = flower[:,np.newaxis].astype(float)

signal = np.repeat(np.mean(observation, axis=0, keepdims=True), repeats=observation.shape[0], axis=0)

noise = observation - signal

transform = transforms.RandomCrop(64)
train_loader, val_loader, n_mean, n_std, s_mean, s_std = create_nm_loader(observation, signal, transform, 0.8, 8)

checkpoint_path = '../nm_checkpoint/Convallaria/signal_dependent'
trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, "PixelCNN"),
                     gpus=1 if str(device).startswith("cuda") else 0,
                     max_epochs=10000,
                     callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_nll"),
                                LearningRateMonitor("epoch"),
                                EarlyStopping('val_nll', patience=10000)])

model = PixelCNN(kernel_size = 7,
                 depth = 10, num_filters = 128,
                 num_gaussians = 10,
                 n_mean=n_mean,
                 n_std=n_std,
                 s_mean=s_mean,
                 s_std=s_std).to(device)

trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('../nm_checkpoint/Convallaria/signal_dependent/final_params.ckpt')