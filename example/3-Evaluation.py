# We import all our dependencies.
import numpy as np
import torch
from torchvision import transforms
from tifffile import imread, imsave
import sys
sys.path.append('../')
from HDN.models.lvae import LadderVAE
from utils import predict_and_save
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#num_obs = 50
observation= torch.from_numpy(imread('../data/Convallaria/observation.tif')).type(torch.float)
signal = torch.mean(observation, axis=0)

observation_cropped = torch.zeros(64*10,1,128,128)
signal_cropped = torch.zeros_like(observation_cropped)

counter = 0
for i in range(10):
    for j in range(8):
        for k in range(8):
            observation_cropped[counter] = observation[i, 0, (j*128):((j+1)*128), (k*128):((k+1)*128)]
            signal_cropped[counter] = signal[0, (j*128):((j+1)*128), (k*128):((k+1)*128)]
            counter += 1

del(observation)
del(signal)

HDN = LadderVAE.load_from_checkpoint('../dn_checkpoint/Convallaria/PixelHDN/final_params.ckpt').to(device)

predict_and_save.predict_and_save(HDN, observation_cropped, '../results/Convallaria', 100)

imsave('../results/Convallaria/signal.tif', signal_cropped.numpy())