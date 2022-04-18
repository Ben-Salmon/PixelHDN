# PixelHDN
Hierarchical DivNoising with autoregressive noise model

data folder contains a link to a google drive from which signal.tif can be downloaded.
The simulated noise is added to the signal to create observations at the start of each
example script.

denoisers folder contains the code for hierarchical divnoising and the original divnoising.

example folder contains example scripts for: training the nosie model, sampling from the
noise model, training the denoiser and predicting results from the denoiser. There are
also two scripts called trainDenoiser_original and predictResults original that train
and sample from HDN how Mangal does in the original repository, without pytorch 
lightning.

jobs folder contains shell scripts for running each of the examples on BEAR.

noise_models folder contains each of the different autoregressive noise models we've 
made: ShiftNet, PixelCNN which uses masked convolutions and gated activation functions
which can be made signal dependent, gaussianRFNoiseModel and AutoRegressiveGMM. 

Currently, example scripts are designed to run ShiftNet and HDN.
