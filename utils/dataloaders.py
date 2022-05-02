import torch
import numpy as np

class nm_dataset(torch.utils.data.Dataset):
    def __init__(self, x_data, s_data=None, transform=None):
        
        self.x_data = torch.from_numpy(x_data).type(torch.float)
        self.s_data = s_data
        if self.s_data is not None:
            self.s_data = torch.from_numpy(self.s_data).type(torch.float)
        
        self.transform = transform
        
        if self.x_data.dim() == 3:
            self.x_data = self.x_data[:,np.newaxis,...]
        elif self.x_data.dim() != 4:
            print('Data dimensions should be [B,C,H,W] or [B,H,W]')
            
        if self.s_data is not None:
            if self.s_data.dim() == 3:
                self.s_data = self.s_data[:,np.newaxis,...]
            elif self.s_data.dim() != 4:
                print('Data dimensions should be [B,C,H,W] or [B,H,W]')
    
    def getparams(self):
        return torch.mean(self.x_data), torch.std(self.x_data)
    
    def __len__(self):
        return self.x_data.shape[0]
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        if self.s_data is not None:
            s = self.s_data[idx]
        
        if self.transform:
            seed = torch.randint(0, 10000, (1,1))
            torch.manual_seed(seed)
            x = self.transform(x)
            if self.s_data is not None:
                torch.manual_seed(seed)
                s = self.transform(s)
                
        if self.s_data is not None:
            return x, s
        else:
            return x

def create_nm_loader(x_data, s_data=None, transform=None, split=0.8, batch_size=32):
    '''
    Creates PyTorch dataloaders for training the noise model.
    Can be loaded with only noise, in which case s_data should be 'None',
    or with corresponding observations and signals. The noise model should
    then subtract the signal from the observation to obtain noise. 
    
    x_data: numpy array, noisy observations or just noise.
    s_data: numpy array or None, clean signals, should be in the same
            order as their noisy counterparts in x_data.
    transform: torchvision transformation.
    split: 0<split<1, portion of dataset to be used in training set.
    batch_size: int, loader batch size.
    '''
    dataset = nm_dataset(x_data, s_data, transform)
    
    data_mean, data_std = dataset.getparams()
    
    train_set, val_set = torch.utils.data.random_split(dataset, [round(len(dataset)*split), round(len(dataset)*(1-split))])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, data_mean, data_std

class dn_dataset(torch.utils.data.Dataset):
    def __init__(self, x_data, transform=None):
        
        self.x_data = torch.from_numpy(x_data).type(torch.float)
        
        self.transform = transform
        
        if self.x_data.dim() == 3:
            self.x_data = self.x_data[:,np.newaxis,...]
        elif self.x_data.dim() != 4:
            print('Data dimensions should be [B,C,H,W] or [B,H,W]')
    
    def getparams(self):
        return torch.mean(self.x_data), torch.std(self.x_data)
    
    def getimgshape(self):
        img = self.__getitem__(0)
        return (img.shape[1], img.shape[2])
    
    def __len__(self):
        return self.x_data.shape[0]
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        
        if self.transform:
            x = self.transform(x)
                
        return x

def create_dn_loader(x_data, transform=None, split=0.8, batch_size=32):
    '''
    Creates PyTorch dataloaders for training the denoiser.
    
    x_data: numpy array, noisy observations.
    transform: torchvision transformation.
    split: 0<split<1, portion of dataset to be used in training set.
    batch_size: int, loader batch size.
    '''
    dataset = dn_dataset(x_data, transform)
    
    data_mean, data_std = dataset.getparams()
    img_shape = dataset.getimgshape()
    
    train_set, val_set = torch.utils.data.random_split(dataset, [round(len(dataset)*split), round(len(dataset)*(1-split))])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
        
    return train_loader, val_loader, data_mean, data_std, img_shape