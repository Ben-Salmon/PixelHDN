import torch
from tqdm import tqdm
from tifffile import imsave
from numpy import mean
import os


def predict_and_save(model, observations, result_dir, num_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.mode_pred = True
    model.eval()
    
    num_obs = observations.shape[0]
    img_shape = (observations.shape[2], observations.shape[3])

    observations = ((observations - model.data_mean) / model.data_std).to(device)
    
    samples = torch.zeros((num_obs,num_samples,img_shape[0],img_shape[1]))

    for i in tqdm(range(num_obs)):
        for j in range(num_samples):
            samples[i, j] = model(observations[i:i+1])['out_mean'].detach()
        
    samples = (samples.cpu() * model.data_std + model.data_mean).numpy()
    
    mmse = mean(samples, axis=1, keepdims=True)
    
    observations = observations * model.data_std + model.data_mean
    
    imsave(os.path.join(result_dir, 'samples.tif'), samples)
    imsave(os.path.join(result_dir, 'observation.tif'), observations.detach().cpu().numpy())
    imsave(os.path.join(result_dir, 'mmse.tif'), mmse)