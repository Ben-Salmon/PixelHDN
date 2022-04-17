import warnings
warnings.filterwarnings('ignore')
# We import all our dependencies.
import numpy as np
import torch
import sys
sys.path.append('/rds/projects/k/krullaff-ibin/ben/PixelHDN16-04/PixelHDN16-04')
from denoisers.HDN.models.lvae import LadderVAE
from denoisers.HDN.boilerplate import boilerplate
import denoisers.HDN.lib.utils as utils
from denoisers.HDN import training
from noise_models.ShiftNet import ShiftNet
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from tifffile import imread
from tqdm import tqdm


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
observation = np.squeeze(signal[:2450] + simdata, 1)

train_data = observation[:int(0.85*observation.shape[0])]
val_data= observation[int(0.85*observation.shape[0]):]

patch_size = 64

img_width = observation.shape[2]
img_height = observation.shape[1]
num_patches = int(float(img_width*img_height)/float(patch_size**2)*1)
train_images = utils.extract_patches(train_data, patch_size, num_patches)
val_images = utils.extract_patches(val_data, patch_size, num_patches)
test_images = val_images[:100]
img_shape = (train_images.shape[1], train_images.shape[2])
print("Shape of training images:", train_images.shape, "Shape of validation images:", val_images.shape)

model_name = "HDN"
directory_path = "dn_checkpoints/Trained_model/" 

noiseModel = ShiftNet.load_from_checkpoint('nm_checkpoint/noiseModel.ckpt').to(device)

# Training-specific
batch_size=64
virtual_batch = 8
lr=3e-4
max_epochs = 500
steps_per_epoch = 100
test_batch_size=100

# Model-specific
num_latents = 6
z_dims = [32]*int(num_latents)
blocks_per_layer = 5
batchnorm = True
free_bits = 1.0
use_uncond_mode_at=[0,1]

train_loader, val_loader, test_loader, data_mean, data_std = boilerplate._make_datamanager(train_images,val_images,
                                                                                           test_images,batch_size,
                                                                                           test_batch_size)

model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel,
                  device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=img_shape,
                  use_uncond_mode_at=use_uncond_mode_at).cuda()

model.train() # Model set in training mode

training.train_network(model=model,lr=lr,max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,
                           directory_path=directory_path,train_loader=train_loader,val_loader=val_loader,
                           test_loader=test_loader,virtual_batch=virtual_batch,
                           gaussian_noise_std=None,model_name=model_name, val_loss_patience=30)