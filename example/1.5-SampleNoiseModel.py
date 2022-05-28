import torch
from tifffile import imread, imsave
import sys
sys.path.append('../')
from noise_models.PixelCNN_signaldependent import PixelCNN
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = PixelCNN.load_from_checkpoint('../nm_checkpoint/Convallaria/signal_dependent/final_params.ckpt').to(device).eval()

signal = imread('../data/Convallaria/signal.tif')[:1]
noise_sample = model.sample(signal))

imsave('../data/noise_model_samples.tif', noise_sample.cpu().numpy())