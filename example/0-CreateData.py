from tifffile import imread, imsave
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
sys.path.append('../')

signal = imread('../data/signal.tif')

noise = (np.random.randn(2486,256,256)).astype(np.float32)
noise = gaussian_filter(noise, (0,0,10))*50
noise = noise + np.random.randn(2486,256,256)*20
noise = noise-noise.mean()

observation = signal + noise

noise = observation - signal

imsave('../data/observation.tif', observation)
imsave('../data/observation_sample.tif', observation[2])
imsave('../data/signal_sample.tif', signal[2])
imsave('../data/noise.tif', noise)
imsave('../data/noise_sample.tif', noise[2])

del(observation)
del(noise)

calibration_noise = (np.random.randn(2486,256,256)).astype(np.float32)
calibration_noise = gaussian_filter(calibration_noise, (0,0,10))*50
calibration_noise = calibration_noise + np.random.randn(2486,256,256)*20
calibration_noise = calibration_noise-calibration_noise.mean()

calibration_data = signal[0] + calibration_noise
imsave('../data/calibration_data.tif', calibration_data)
imsave('../data/calibration_data_sample.tif', calibration_data[3])