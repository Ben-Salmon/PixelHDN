from tifffile import imread
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
sys.path.append('../')
from denoisers.divnoising.nets import lightningmodel
from denoisers.divnoising.divnoising import utils

# Load best weights
dn_checkpoint_path = '../dn_checkpoint/divnoising_best.ckpt'
vae = lightningmodel.VAELightning.load_from_checkpoint(checkpoint_path = '../dn_checkpoint/divnoising_best.ckpt', strict=False)
vae.to(device)

observation = imread('../data/observation.tif')
data_mean = np.mean(observation)
data_std = np.

## Use the trained denoiser to produce and save some results
num_samples = 100
export_results_path = '../data/results'
fraction_samples_to_export = 1
export_mmse = True
tta = False   # 8-fold test-time augmentation
mmse_results = utils.predict_and_save(observation[:15,...,:,:],vae,num_samples,device,fraction_samples_to_export,export_mmse,export_results_path,tta)
'''
# We import all our dependencies.
import numpy as np
import torch
from torchvision import transforms

from tifffile import imread, imsave

import sys
sys.path.append('/rds/projects/k/krullaff-ibin/ben/HDN/')

from models.lvae import LadderVAE

from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

observation= torch.from_numpy(imread('data/observation.tif')[:,np.newaxis,...]).type(torch.float)
signal = torch.from_numpy(imread('data/signal.tif'))

mean = torch.mean(observation)
std = torch.std(observation)

observation0 = (observation[3:4] - mean) / std
target = signal[3:4]

del(observation)
del(signal)

observation0 = transforms.CenterCrop(64)(observation0).to(device)
target = transforms.CenterCrop(64)(target)

HDN = LadderVAE.load_from_checkpoint('dn_checkpoint/final_params.ckpt').to(device)
HDN.mode_pred = True
HDN.eval()

num_samples = 10 # number of samples used to compute MMSE estimate
samples = torch.zeros((1,10,64,64))

for i in tqdm(range(observation0.shape[0])):
    for j in range(num_samples):
        samples[i, j] = HDN(observation0[i:i+1])['out_mean']
        
samples = samples * std + mean
observation0 = observation0 * std + mean
imsave('data/results/samples.tif', samples.detach().numpy())
imsave('data/results/target.tif', target.numpy())
imsave('data/results/observation.tif', observation0.detach().cpu().numpy())
'''