# Decoding natural image stimuli from fMRI data with a surface-based convolutional network

Code for MIDL 2023 paper [Decoding natural image stimuli from fMRI data with a surface-based convolutional network](https://arxiv.org/abs/2212.02409) (oral).

![](/model.png)

## Requirements
1. Clone the IC-GAN repo: `git clone https://github.com/facebookresearch/ic_gan.git`
2. Install required packages: 
```shell
conda env create -f environment.yml 
conda activate meshconvdec
```
3. If you use NSD, it is publicly available at https://naturalscenesdataset.org/. Follow the 2nd step in the Instructions to prepare the data.

## Instructions
1. Install required packages.
2. (optional) If the data is in fsaverage space instead of fs_LR_32K surface, e.g., the data in NSD repo here https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/. Run `loop4all.sh` which will run`map_fsaverage_to_hcp.sh` for every session and every subject in NSD.
2. Run `python train_feature_decoding.py` to train the `Cortex2Semantic` model.
3. Run `python train_combined_decoding.py` to train the `Cortex2Detail` model.
4. Run `python test.py` to generate the decoded images. 

Please note that the file paths and the hyparameters may need to be changed according to your own settings.

## Availability
We welcome researchers to use our models and to compare their new approaches with ours.
Pretrained models and reconstructed images for 1000 shared images in NSD can be downloaded [here](https://cornell.box.com/s/epev6y4y6foqjey4pxtg4txsyfmcvwmj).

## Citation
If you find this work helpful for your research, please cite our paper:
```
@article{gu2022decoding,
  title={Decoding natural image stimuli from fMRI data with a surface-based convolutional network},
  author={Gu, Zijin and Jamison, Keith and Kuceyeski, Amy and Sabuncu, Mert},
  journal={arXiv preprint arXiv:2212.02409},
  year={2022}
}
```



