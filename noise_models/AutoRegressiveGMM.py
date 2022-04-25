import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
import sys

class AutoRegressiveGMM(pl.LightningModule):
    def __init__(self, mean, std, num_gaussians):
        super().__init__()  
        self.mean = mean
        self.std = std
        self.num_gaussians = num_gaussians
        
    
    
    def get_gaussian_params(self, pred):
        blockSize = self.num_gaussians
        means = pred[:,:blockSize,...]
        stds = torch.sqrt(torch.exp(pred[:,blockSize:2*blockSize,...]))
        weights = torch.exp(pred[:,2*blockSize:3*blockSize,...])
        weights = weights / torch.sum(weights,dim = 1, keepdim = True)
        return means, stds, weights
    
    def normalDens(self, x, mu, sigma):
        pi = torch.tensor([np.pi]).to(self.device)
        p = (x-mu)**2
        p = p / (2.0*sigma**2)
        p = torch.exp(p)
        p = p / torch.sqrt((2.0*pi)*sigma**2)
        return torch.log(p)
    
    def likelihood(self, x, means, stds, weights, n_gaussians):
        p=torch.zeros_like(x)
        p=torch.repeat_interleave(x, n_gaussians, dim=1)
        for gaussian in range(n_gaussians):
            p=self.normalDens(x, means[:,gaussian], stds[:,gaussian]) * weights[:,gaussian]
        return p+1e-10
       
    def vaeloglikelihood(self, x, s=None):
        if s is None:
            s = self.mean
        # Separate noise from signal and normalise
        
        x = x - self.mean
        s = s - self.mean
        
        x = x / self.std
        s = s / self.std
        
        n = x - s
        
        if self.training:
            pred = self.forward(n)
        else:
            pred = self.forward(n).detach()
            
        means, stds, weights = self.get_gaussian_params(pred)
        likelihoods= 0.5*((means-n)/stds)**2 - torch.log(stds) -np.log(2.0*np.pi)*0.5
        temp = torch.max(likelihoods, dim = 1, keepdim = True)[0].detach()
        likelihoods=torch.exp( likelihoods -temp) * weights
#        likelihoods=torch.exp( likelihoods) * weights
        loglikelihoods = torch.log(torch.sum(likelihoods, dim = 1, keepdim = True))
        loglikelihoods = loglikelihoods + temp 
        return loglikelihoods    
    
    def loglikelihood(self, x, s=None):
        if s is None:
            s = self.mean
        # Separate noise from signal and normalise
        
        x = x - self.mean
        s = s - self.mean
        
        x = x / self.std
        s = s / self.std
        
        n = x - s
        
        if self.training:
            pred = self.forward(n)
        else:
            pred = self.forward(n).detach()
            
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
    def sample(self, img_shape, img=None):
        """Sampling function for the autoregressive model.

        Args:
            img_shape: Shape of the image to generate (B,C,H,W)
            img (optional): If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = torch.zeros(img_shape, dtype=torch.float).to(self.device) - 1
        # Generation loop
        with tqdm(total=img_shape[2], file=sys.stdout) as pbar:
            for h in range(img_shape[2]):
                pbar.set_description('done: %d' % (h + 1))
                pbar.update(1)
                for w in range(img_shape[3]):
                    for c in range(img_shape[1]):
                        # Skip if not to be filled (-1)
                        if (img[:, c, h, w] != -1).all().item():
                            continue
                        # For efficiency, we only have to input the upper part of the image
                        # as all other parts will be skipped by the masked convolutions anyway
                        pred = self.forward(img[:, :, : h + 1, :]).detach()
                        means, stds, weights = self.get_gaussian_params(pred)
                        means = means[:,:,h,w][...,np.newaxis,np.newaxis]
                        stds = stds[:,:,h,w][...,np.newaxis,np.newaxis]
                        weights = weights[:,:,h,w][...,np.newaxis,np.newaxis]
                        samp = self.sampleFromMix(means, stds, weights).detach()
                        img[:, c, h, w] = samp[:,0,0]

        return img*self.std +self.mean
    
    def training_step(self, batch, batch_idx):
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("train_nll", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("val_nll", loss)

    def test_step(self, batch, batch_idx):
        (x, s) = batch
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("test_nll", loss)
