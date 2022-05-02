import warnings
warnings.filterwarnings('ignore')
import os
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

observation = imread('../data/observation.tif')

transform = transforms.RandomCrop(64)
train_loader, val_loader, data_mean, data_std, img_shape = create_dn_loader(observation, transform=transform)
del(observation)

noiseModel = PixelCNN.load_from_checkpoint('../nm_checkpoint/final_params.ckpt').to(device).eval()
for param in noiseModel.parameters():
    param.requires_grad = False
    
lr=3e-4
num_latents = 5
z_dims = [32]*int(num_latents)
free_bits = 1.0

model = LadderVAE(z_dims=z_dims,               
                  data_mean=data_mean,
                  data_std=data_std,
                  noiseModel=noiseModel,
                  free_bits=free_bits,
                  img_shape=img_shape,
                  img_folder='../dn_checkpoint/imgs',
                  lr=lr).to(device)

checkpoint_path = '../dn_checkpoint'
trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, "HDN"),
                     gpus=1 if str(device).startswith("cuda") else 0,
                     max_epochs=2000,
                     logger=False,
                     gradient_clip_val=0,
                     callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_elbo"),
                                EarlyStopping('val_elbo', patience=30)])

trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('../dn_checkpoint/final_params.ckpt')