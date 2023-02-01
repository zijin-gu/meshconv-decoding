# Decoding natural image stimuli from fMRI data with a surface-based convolutional network

Code for paper [Decoding natural image stimuli from fMRI data with a surface-based convolutional network](https://arxiv.org/abs/2212.02409).

![](/model.png)

## Requirements
1. Clone the IC-GAN repo: `git clone https://github.com/facebookresearch/ic_gan.git`
2. Install required packages: 
```shell
conda env create -f environment.yml 
conda activate meshconvdec
```

## Insrtuctions
1. install required packages.
2. (optional) if the data is not in fs_LR_32K surface, run `map_fsaverage_to_hcp.sh`.
2. run `python train_feature_decoding.py` to train the `Cortex2Semantic` model.
3. run `python train_combined_decoding.py` to train the `Cortex2Detail` model.
4. run `python test.py` to generate the decoded images. Pretrained models can be downloaded [here](https://cornell.box.com/s/dpzt57eeg3424wd2b330dmttg86x7ffm)

Note that the file paths and the hyparameters may need to be changed according to your own settings.

