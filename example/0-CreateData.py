from tifffile import imsave
from scipy.io import loadmat

fig1 = loadmat('../data/images/fig1.mat')['sensor_data']    
    
noise = fig1[...,500:]
observation = fig1[...,30:500]

imsave('../data/noise.tif', noise)
imsave('../data/observation.tif', observation)
'''

from tifffile import imread, imsave
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
sys.path.append('../')

signal = imread('../data/signal.tif')

noise = (np.random.randn(2486,256,256)).astype(np.float32)
noise = gaussian_filter(noise, (0,2,10))*100
noise = noise + np.random.randn(2486,256,256)*25
noise = noise-noise.mean()

observation = signal + noise

noise = observation - signal

imsave('../data/observation.tif', observation)
imsave('../data/observation_sample.tif', observation[2])
imsave('../data/signal_sample.tif', signal[2])
imsave('../data/noise.tif', noise)
imsave('../data/noise_sample.tif', noise[2])
'''