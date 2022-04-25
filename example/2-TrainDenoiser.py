'''
import warnings
warnings.filterwarnings('ignore')
from tifffile import imread

import torch

import sys
sys.path.append('../')
from noise_models.ShiftNet import ShiftNet
from denoisers.divnoising.divnoising import utils, training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images
observation = imread('../data/observation.tif')

# Load trained noise model
noiseModel = ShiftNet.load_from_checkpoint('../nm_checkpoint/final_params.ckpt').to(device).eval()
for param in noiseModel.parameters():
    param.requires_grad = False

# Create training patches of noisy observations
train_patches, val_patches = utils.get_trainval_patches(observation,augment=False,patch_size=64,num_patches=10)
x_train_tensor, x_val_tensor, data_mean, data_std = utils.preprocess(train_patches, val_patches)
# Define hyperparameters
dn_checkpoint_path = '../dn_checkpoint'
n_depth=2
batch_size=64
max_epochs=int(22000000/(x_train_tensor.shape[0]))
model_name = 'divnoising' # a name used to identify the model
real_noise=True
# Train the denoiser
training.train_network(x_train_tensor, x_val_tensor, batch_size, data_mean, data_std, 
                       None, noiseModel, n_depth=n_depth, max_epochs=max_epochs, 
                       model_name=model_name, basedir=dn_checkpoint_path, log_info=True)

'''
import warnings
warnings.filterwarnings('ignore')
# We import all our dependencies.
import numpy as np
import torch
import sys
sys.path.append('../')
from denoisers.HDN.models.lvae import LadderVAE
from denoisers.HDN.boilerplate import boilerplate
import denoisers.HDN.lib.utils as utils
from denoisers.HDN import training
from tifffile import imread
from noise_models.ShiftNet import ShiftNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

observation = imread('../data/observation.tif')
data_mean = np.mean(observation)
data_std = np.std(observation)

train_data = observation[:int(0.85*observation.shape[0])]
val_data= observation[int(0.85*observation.shape[0]):]
print("Shape of training images:", train_data.shape, "Shape of validation images:", val_data.shape)

### We extract overlapping patches of size ```patch_size x patch_size``` from training and validation images.
### Usually 64x64 patches work well for most microscopy datasets
patch_size = 64

img_width = observation.shape[2]
img_height = observation.shape[1]
num_patches = int(float(img_width*img_height)/float(patch_size**2)*1)
train_images = utils.extract_patches(train_data, patch_size, num_patches)
val_images = utils.extract_patches(val_data, patch_size, num_patches)
val_images = val_images[:1000] # We limit validation patches to 1000 to speed up training but it is not necessary
test_images = val_images[:100]
img_shape = (train_images.shape[1], train_images.shape[2])
print("Shape of training images:", train_images.shape, "Shape of validation images:", val_images.shape)

model_name = "HDN"
directory_path = "../dn_checkpoint/" 

# Data-specific
gaussian_noise_std = None
noiseModel = ShiftNet.load_from_checkpoint('../nm_checkpoint/final_params.ckpt').to(device).eval()
for param in noiseModel.parameters():
    param.requires_grad = False

# Training-specific
batch_size=64
virtual_batch = 8
lr=3e-4
max_epochs = 500
steps_per_epoch=100
test_batch_size=100

# Model-specific
num_latents = 6
z_dims = [32]*int(num_latents)
blocks_per_layer = 5
batchnorm = True
free_bits = 1.0

train_loader, val_loader, test_loader, data_mean, data_std = boilerplate._make_datamanager(train_images,val_images,
                                                                                           test_images,batch_size,
                                                                                           test_batch_size,
                                                                                           data_mean,
                                                                                           data_std)

model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel,
                  device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=img_shape).cuda()

model.train() # Model set in training mode

training.train_network(model=model,lr=lr,max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,directory_path=directory_path,
                       train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,
                       virtual_batch=virtual_batch,gaussian_noise_std=gaussian_noise_std,
                       model_name=model_name,val_loss_patience=10)