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

num_obs = 400
observation= torch.from_numpy(imread('../data/observation.tif')[:,np.newaxis,...]).type(torch.float)

observation = observation[:num_obs]

HDN = LadderVAE.load_from_checkpoint('../dn_checkpoint/MRI/PixelHDN/HDN/checkpoints/epoch=1699-step=105399.ckpt').to(device)

predict_and_save.predict_and_save(HDN, observation, '../results/MRI/PixelHDN', 100)