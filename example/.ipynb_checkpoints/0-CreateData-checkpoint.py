from tifffile import imsave
from scipy.io import loadmat
import numpy as np

fig1 = loadmat('../data/images/fig1.mat')['sensor_data']
    
noise = np.transpose(fig1[:128,:128,500:], (2, 0, 1))
observation = np.transpose(fig1[:128,:128,32:500], (2, 0, 1))

imsave('../data/PAI/noise.tif', noise)
imsave('../data/PAI/observation.tif', observation)
'''
from tifffile import imread, imsave
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
sys.path.append('../')

noise = (np.random.randn(2486,256,256)).astype(np.float32)
noise = gaussian_filter(noise, (0,1,5))*110
noise = noise + np.random.randn(2486,256,256)*20
noise = noise-noise.mean()

signal = imread('../data/signal.tif')

observation = signal + noise

imsave('../data/observation.tif', observation)
imsave('../data/noise.tif', noise)
'''