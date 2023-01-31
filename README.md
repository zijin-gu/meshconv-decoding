# Decoding natural image stimuli from fMRI data with a surface-based convolutional network

Code for paper [Decoding natural image stimuli from fMRI data with a surface-based convolutional network](https://arxiv.org/abs/2212.02409).

![](/model.png)

## Requirements
1. Clone the IC-GAN repo: https://github.com/facebookresearch/ic_gan.git
2. Other packages required:

 `argparse`
`numpy`
`os` 
`torch`
`torchvision`
`h5py`
`math`
`pickle`
`scipy`
`sys`
`collections`

## Insrtuctions
1. install required packages.
2. (optional) if the data is not in fs_LR_32K surface, run `map_fsaverage_to_hcp.sh`
2. run `python train_feature_decoding.py` to train the `Cortex2Semantic` model.
3. run `python train_combined_decoding.py` to train the `Cortex2Detail` model.
