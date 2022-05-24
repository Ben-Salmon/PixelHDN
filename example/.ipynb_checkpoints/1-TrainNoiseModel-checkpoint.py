import os
import torch
from tifffile import imread
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import sys
sys.path.append('../')
from noise_models.PixelCNN import PixelCNN
from utils.dataloaders import create_nm_loader
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noise = imread('../data/Convallaria/noise.tif')

transform = transforms.RandomCrop(64)
train_loader, val_loader, data_mean, data_std = create_nm_loader(noise, batch_size=8, split=0.9, transform=transform)

checkpoint_path = '../nm_checkpoint/Convallaria'
trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, "PixelCNN"),
                     gpus=1 if str(device).startswith("cuda") else 0,
                     max_epochs=2000,
                     callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_nll"),
                                LearningRateMonitor("epoch"),
                                EarlyStopping('val_nll', patience=2000)])

model = PixelCNN(kernel_size = 7,
                 depth = 5, num_filters = 128,
                 num_gaussians = 10,
                 data_mean = data_mean,
                 data_std = data_std).to(device)

trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('../nm_checkpoint/Convallaria/final_params.ckpt')