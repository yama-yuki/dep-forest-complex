#!/bin/bash

## paths
PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
RESCORE_DIR=${PJ_DIR}/rescore_module
rescore_config=${RESCORE_DIR}/rescore.cfg

## EisnerK
EisnerK=4
PKL_DIR=${PJ_DIR}/biaffine_forest/pkl/k${EisnerK}

## parameters
K=4
alpha=0.7
beta=0.5

## mode
rescore=True
test=False

if [ ${rescore} = True ] ; then
a=${alpha//'.'/''}
b=${beta//'.'/''}
out_name=rescore_${K}-${a}-${b};
else
out_name=vanilla_${K}
fi

echo ${out_name}

## out_path
OUT_DIR=${PJ_DIR}/outputs/k${EisnerK}
mkdir -p ${OUT_DIR}
out_path=${OUT_DIR}/${out_name}.conllu

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

