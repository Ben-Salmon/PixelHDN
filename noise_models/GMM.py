import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
import sys

class GMM(pl.LightningModule):
    def __init__(self, mean, std, num_gaussians, signal_dep):
        super().__init__()  
        self.mean = mean
        self.std = std
        self.num_gaussians = num_gaussians
        self.signal_dep = signal_dep
        
    def get_gaussian_params(self, pred):
        blockSize = self.num_gaussians
        means = pred[:,:blockSize,...]
        stds = torch.sqrt(torch.exp(pred[:,blockSize:2*blockSize,...]))
        weights = torch.exp(pred[:,2*blockSize:3*blockSize,...])
        weights = weights / torch.sum(weights,dim = 1, keepdim = True)
        return means, stds, weights
    
    def loglikelihood(self, x, s=None):     
        if s is None:
            s = self.mean
            
        x = (x - self.mean)/self.std
        s = (s - self.mean)/self.std
        n = x - s
        
        pred = self.forward(n, s)
            
        means, stds, weights = self.get_gaussian_params(pred)
        likelihoods= -0.5*((means-n)/stds)**2 - torch.log(stds) -np.log(2.0*np.pi)*0.5
        temp = torch.max(likelihoods, dim = 1, keepdim = True)[0].detach()
        likelihoods=torch.exp( likelihoods -temp) * weights
#        likelihoods=torch.exp( likelihoods) * weights
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
    def sample(self, s):
        """Sampling function for the autoregressive model.

        Args:
            img_shape: Shape of the image to generate (B,C,H,W)
            img (optional): If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        n = torch.zeros(s.shape, dtype=torch.float).to(self.device) - 1
        # Generation loop
        with tqdm(total=s.shape[2], file=sys.stdout) as pbar:
            for h in range(s.shape[2]):
                pbar.set_description('done: %d' % (h + 1))
                pbar.update(1)
                for w in range(s.shape[3]):
                    for c in range(s.shape[1]):
                        # For efficiency, we only have to input the upper part of the image
                        # as all other parts will be skipped by the masked convolutions anyway
                        pred = self.forward(n[:, :, : h + 1, :], s[:, :, : h + 1, :]).detach()
                        means, stds, weights = self.get_gaussian_params(pred)
                        means = means[:,:,h,w][...,np.newaxis,np.newaxis]
                        stds = stds[:,:,h,w][...,np.newaxis,np.newaxis]
                        weights = weights[:,:,h,w][...,np.newaxis,np.newaxis]
                        samp = self.sampleFromMix(means, stds, weights).detach()
                        n[:, c, h, w] = samp[:,0,0]

        return n*self.std +self.mean
    
    def training_step(self, batch, batch_idx):
        if self.signal_dep:
            (x, s) = batch
            loss = -torch.mean(self.loglikelihood(x, s))
        else:
            loss = -torch.mean(self.loglikelihood(batch))
        self.log("train_nll", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.signal_dep:
            (x, s) = batch
            loss = -torch.mean(self.loglikelihood(x, s))
        else:
            loss = -torch.mean(self.loglikelihood(batch))
        self.log("val_nll", loss)