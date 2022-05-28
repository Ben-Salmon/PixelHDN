import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
import sys

class GMM(pl.LightningModule):
    def __init__(self, n_mean, n_std, s_mean, s_std, num_gaussians):
        super().__init__()  
        self.n_mean = n_mean
        self.n_std = n_std
        self.s_mean = s_mean
        self.s_std = s_std
        self.num_gaussians = num_gaussians
        
    def get_gaussian_params(self, pred):
        blockSize = self.num_gaussians
        means = pred[:,:blockSize,...]
        stds = torch.sqrt(torch.exp(pred[:,blockSize:2*blockSize,...]))
        weights = torch.exp(pred[:,2*blockSize:3*blockSize,...])
        weights = weights / torch.sum(weights,dim = 1, keepdim = True)
        return means, stds, weights   
    
    def loglikelihood(self, x, s=None):
        if s is None:
            s = torch.zeros_like(x)
        # Separate noise from signal and normalise
        
        n = x-s
        
        n = n - self.n_mean
        n = n / self.n_std

        s =  s - self.s_mean
        s =  s / self.s_std
        
        if self.training:
            pred = self.forward(n, s)
        else:
            pred = self.forward(n, s).detach()
            
        means, stds, weights = self.get_gaussian_params(pred)
        likelihoods= -0.5*((means-n)/stds)**2 - torch.log(stds) -np.log(2.0*np.pi)*0.5
        temp = torch.max(likelihoods, dim = 1, keepdim = True)[0].detach()
        likelihoods=torch.exp( likelihoods -temp) * weights
        loglikelihoods = torch.log(torch.sum(likelihoods, dim = 1, keepdim = True))
        loglikelihoods = loglikelihoods + temp 
        return loglikelihoods
    
    def sampleFromMix(self, means, stds, weights):
        num_components = means.shape[1]
        shape = means[:,0,...].shape
        selector = torch.rand(shape, device = means.device)
        gauss = torch.normal(means[:,0,...]*0, means[:,0,...]*0 + 1)
        out = means[:,0,...]*0

        for i in range(num_components):
            mask = torch.zeros(shape)
            mask = (selector<weights[:,i,...]) & (selector>0)
            out += mask* (means[:,i,...] + gauss*stds[:,i,...])
            selector -= weights[:,i,...]
        
        del gauss
        del selector
        del shape
        return out

    @torch.no_grad()    
    def sample(self, signal):
        """Sampling function for the autoregressive model.

        Args:
            img_shape: Shape of the image to generate (B,C,H,W)
            img (optional): If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        s = (signal - self.s_mean) / self.s_std
        # Create empty image
        n = torch.zeros(s.shape, dtype=torch.float).to(self.device) - 1
        # Generation loop
        with tqdm(total=s.shape[2], file=sys.stdout) as pbar:
            for h in range(s.shape[2]):
                pbar.set_description('done: %d' % (h + 1))
                pbar.update(1)
                for w in range(s.shape[3]):
                    for c in range(s.shape[1]):
                        # Skip if not to be filled (-1)
                        if (n[:, c, h, w] != -1).all().item():
                            continue
                        # For efficiency, we only have to input the upper part of the image
                        # as all other parts will be skipped by the masked convolutions anyway
                        pred = self.forward(n[:, :, : h + 1, :], s[:, :, : h + 1, :]).detach()
                        means, stds, weights = self.get_gaussian_params(pred)
                        means = means[:,:,h,w][...,np.newaxis,np.newaxis]
                        stds = stds[:,:,h,w][...,np.newaxis,np.newaxis]
                        weights = weights[:,:,h,w][...,np.newaxis,np.newaxis]
                        samp = self.sampleFromMix(means, stds, weights).detach()
                        n[:, c, h, w] = samp[:,0,0]

        return n*self.n_std + self.n_mean
    
    
    def training_step(self, batch, batch_idx):
        loss = -torch.mean(self.loglikelihood(batch[0], batch[1]))
        self.log("train_nll", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = -torch.mean(self.loglikelihood(batch[0], batch[1]))
        self.log("val_nll", loss, prog_bar=True)