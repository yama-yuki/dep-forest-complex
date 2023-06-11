#!/bin/bash

## paths
PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
PKL_DIR=${PJ_DIR}/biaffine_forest/pkl
RESCORE_DIR=${PJ_DIR}/rescore_module
rescore_config=${RESCORE_DIR}/rescore.cfg

## parameters
K=4
alpha=1
beta=1

## mode
rescore=True
test=False

if [ ${rescore} = True ] ; then
out_name='' ;
else
out_name='_norescore'
fi

## out_path
out_path=${PJ_DIR}/outputs/pred_${K}_${alpha}${out_name}.conllu

#SBATCH --job-name=forest_parse
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
. ~/.bashrc
conda activate for
conda info -e

cd ${PJ_DIR}
python forest_decoder.py \
    --pkl_dir=${PKL_DIR} \
    --out_path=${out_path} \
    --rescore_cfg=${rescore_config} \
    --K=${K} \
    --alpha=${alpha} \
    --beta=${beta} \
    --rescore=${rescore} \
    --test=${test}

