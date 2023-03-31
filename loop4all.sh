#!/bin/bash

i=$1

mkdir -p ./nsd/responses/subj0$i/fs_LR_32k
for j in {01..40}
do
    bash map_fsaverage_to_hcp.sh ./nsd/responses/subj0$i/fsaverage/betas_fithrf_GLMdenoise_RR/lh.betas_session$j.mgh ./nsd/responses/subj0$i/fs_LR_32k/lh_betas_session$j.func.gii
    bash map_fsaverage_to_hcp.sh ./nsd/responses/subj0$i/fsaverage/betas_fithrf_GLMdenoise_RR/rh.betas_session$j.mgh ./nsd/responses/subj0$i/fs_LR_32k/rh_betas_session$j.func.gii
done
