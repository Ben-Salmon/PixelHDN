#!/bin/bash
#SBATCH --time 12:00:00
#SBATCH --qos castlesgpu
#SBATCH --account krullaff-ibin
#SBATCH --mem 64G
#SBATCH --gres gpu:1

module purge; module load bluebear
module load torchvision/0.11.1-foss-2021a-CUDA-11.3.1
module load Cellpose/0.6.5-foss-2021a-CUDA-11.3.1
module load pytorch-lightning/1.5.7-foss-2021a-CUDA-11.3.1

python3 example/predictResults.py
