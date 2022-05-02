# We import all our dependencies.
import numpy as np
import torch
from torchvision import transforms
from tifffile import imread
import sys
sys.path.append('../')
from HDN.models.lvae import LadderVAE
from utils import predict_and_save
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_obs = 100

observation= torch.from_numpy(imread('../data/observation.tif')[:,np.newaxis,...]).type(torch.float)
observation0 = observation[:num_obs]
observation0 = transforms.CenterCrop(64)(observation0).to(device)

mean = torch.mean(observation)
std = torch.std(observation)

observation0 = observation[:num_obs]
observation0 = transforms.CenterCrop(64)(observation0).to(device)

HDN = LadderVAE.load_from_checkpoint('../dn_checkpoint/final_params.ckpt').to(device)

predict_and_save.predict_and_save(HDN, observation0, '../data/results/PAI 2', 100, mean, std)