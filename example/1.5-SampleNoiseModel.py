import torch
from tifffile import imsave
import sys
sys.path.append('../')
from noise_models.ShiftNet import ShiftNet
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = ShiftNet.load_from_checkpoint('../nm_checkpoint/final_params.ckpt').to(device).eval()

noise_sample = model.sample(s=torch.zeros(1,1,64,64))

imsave('../data/noise_model_samples.tif', noise_sample.cpu().numpy())
