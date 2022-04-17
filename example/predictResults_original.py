# We import all our dependencies.
import numpy as np
import torch
from torchvision import transforms

import sys
sys.path.append('/rds/projects/k/krullaff-ibin/ben/PixelHDN16-04/PixelHDN16-04')

from denoisers.HDN.models.lvae import LadderVAE
from denoisers.HDN.boilerplate import boilerplate
import denoisers.HDN.lib.utils as utils

from tifffile import imread, imsave

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import torch.utils.data as dt


simdata = (np.random.randn(36,256,256)).astype(np.float32)
simdata = gaussian_filter(simdata, (0,0,4))
simdata = simdata + np.random.randn(36,256,256)*0.1
simdata = simdata-simdata.mean()
simdata = simdata[:,np.newaxis,...]*150.0
simdata = simdata - simdata.mean()
dataStd = np.std(simdata)

signal = np.squeeze(imread('data/signal.tif'), 1)[2450:]
observation = signal + np.squeeze(simdata, 1)
signal = transforms.RandomCrop(64)(torch.from_numpy(signal)).numpy()
observation = transforms.RandomCrop(64)(torch.from_numpy(observation)).numpy()

img_width, img_height = signal.shape[1], signal.shape[2]

model = torch.load("/rds/projects/k/krullaff-ibin/ben/PixelHDN16-04/PixelHDN16-04/dn_checkpoints/Trained_model/model/HDN_best_vae.net")
model.mode_pred=True
model.eval()

gaussian_noise_std = None
num_samples = 100 # number of samples used to compute MMSE estimate
tta = False # turn on test time augmentation when set to True. It may improve performance at the expense of 8x longer prediction time
psnrs = []
img_mmse = np.zeros_like(observation)
samples = np.zeros((observation.shape[0], num_samples, observation.shape[1], observation.shape[2]))
range_psnr = np.max(signal[0])-np.min(signal[0])
for i in range(observation.shape[0]):
    img_mmse[i], samples[i] = boilerplate.predict(observation[i],num_samples,model,gaussian_noise_std,device,tta)
    psnr = utils.PSNR(signal[0], img_mmse[i], range_psnr)
    psnrs.append(psnr)
    print("image:", i, "PSNR:", psnr, "Mean PSNR:", np.mean(psnrs))

imsave('results/mmse.tif', img_mmse)
imsave('results/samples.tif', samples)