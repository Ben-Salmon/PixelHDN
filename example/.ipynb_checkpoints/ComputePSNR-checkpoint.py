import numpy as np
from tifffile import imread

def PSNR(gt, img, psnrRange):
    '''
    Compute PSNR.
    Parameters
    ----------
    gt: array
        Ground truth image.
    img: array
        Predicted image.
    psnrRange: float
        Range PSNR
    '''
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(psnrRange) - 10 * np.log10(mse)

result = imread('../results/MRI/PixelHDN/mmse.tif')
signal = imread('../data/MRI/signal.tif')

psnr = np.array([])
for i in range(result.shape[0]):
    range_psnr = np.max(signal[i])-np.min(signal[i])
    psnr = np.append(psnr, PSNR(signal[i], result[i], range_psnr))

print(np.mean(psnr))