import torch
dtype = torch.float
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

def fastShuffle(series, num):
    length = series.shape[0]
    for i in range(num):
        series = series[np.random.permutation(length),:]
    return series

#import divnoising.histNoiseModel

class GaussianRFNoiseModel:

    def __init__(self, f_h, f_v, f_d, size):        
        
        self.f_h = f_h
        self.f_v = f_v
        self.f_d = f_d
        
        self.size = size
        sz = size*size
        x = torch.arange(sz)
        y = torch.arange(sz)
        xv, yv = torch.meshgrid([x, y])
        self.prec = torch.eye(sz) * f_d
        self.prec[(xv==yv+1) & (xv%size != 0)] = f_h
        self.prec[(xv==yv-1) & (yv%size != 0)] = f_h
        self.prec[(xv+size==yv)] = f_v
        self.prec[(xv-size==yv)] = f_v

    ## CUSTOM FUNCTIONS ##
    def createnoise(self, n):
    # Simulates noisy observations from signals
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        prec = self.prec.to(device)
        
        x = MultivariateNormal(loc=torch.zeros(self.size**2, device=device), precision_matrix=prec).rsample(torch.Size([n]))
        return torch.reshape(x, (n, self.size, self.size)).cpu().detach().numpy()

    def likelihood(self, observations, signals):
        
        noise = observations - signals
        
        pairwiseh = torch.sum(noise[...,1:] * noise[...,:-1]) * self.f_h * 2
        pairwisev = torch.sum(noise[...,1:,:] * noise[...,:-1,:]) * self.f_v * 2
        squared = torch.sum(noise**2) * self.f_d
        #print(self.f_h,self.f_v )
        return -0.5 * (pairwiseh + pairwisev + squared)