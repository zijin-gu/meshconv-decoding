#!/bin/bash

set -e

infile=$1
outgii=$2 #must be "*.func.gii"

if [[ ${outgii} != *.func.gii ]]; then
    echo "Output file must end in func.gii!"
    exit 1
fi

#downloaded from: https://github.com/Washington-University/HCPpipelines/tree/master/global/templates/standard_mesh_atlases
templatedir=$HOME/HCPpipelines/global/templates/standard_mesh_atlases/resample_fsaverage

#lh.* -> L, rh.* -> R
#H=$(basename ${infile} | sed -E 's/^lh\..+$/L/g' | sed -E 's/^rh\..+$/R/')
H=$(basename ${infile} | sed "s/^[^.]*\.\([^.]*\)\..*/\1/g")

tmpfile=${infile}_tmp.func.gii

#use freesurfer mri_convert to go from .mgh to .gii
mri_convert ${infile} ${tmpfile}

#use HCP connectome workbench wb_command to map to fs_LR32k
wb_command -metric-resample ${tmpfile} $templatedir/fsaverage_std_sphere.$H.164k_fsavg_$H.surf.gii $templatedir/fs_LR-deformed_to-fsaverage.$H.sphere.32k_fs_LR.surf.gii ADAP_BARY_AREA ${outgii} -area-metrics $templatedir/fsaverage.$H.midthickness_va_avg.164k_fsavg_$H.shape.gii $templatedir/fs_LR.$H.midthickness_va_avg.32k_fs_LR.shape.gii
# wb_command -metric-resample ${tmpfile} $templatedir/fsaverage6_std_sphere.$H.41k_fsavg_$H.surf.gii $templatedir/fs_LR-deformed_to-fsaverage.$H.sphere.32k_fs_LR.surf.gii ADAP_BARY_AREA ${outgii} -area-metrics $templatedir/fsaverage6.$H.midthickness_va_avg.41k_fsavg_$H.shape.gii $templatedir/fs_LR.$H.midthickness_va_avg.32k_fs_LR.shape.gii

rm -f ${tmpfile}
