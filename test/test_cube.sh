#!/bin/bash

model_type=V-V
data_type=test
if [ ${data_type} = mytree ] ; then
data=mytree_upos;
else
data=test
fi
echo ${model_type}
echo ${data_type}
echo ${data}

## paths
PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
RESCORE_DIR=${PJ_DIR}/rescore_module
rescore_config=${RESCORE_DIR}/rescore.cfg

## EisnerK
EisnerK=8
PKL_DIR=${PJ_DIR}/biaffine_forest/pkl/${data_type}/k${EisnerK}

## parameters
K=8
alpha=0.3
beta=0.1

## mode
rescore=False
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
OUT_DIR=${PJ_DIR}/outputs/${model_type}/${data_type}/k${EisnerK}
mkdir -p ${OUT_DIR}
out_path=${OUT_DIR}/${out_name}.conllu

#sbatch --gres=gpu:1 --partition=gpu_long --job-name=cube_pruning --time=96:00:00 test_cube.sh
. ~/.bashrc
conda activate for
conda info -e

cd ${PJ_DIR}
python forest_rescorer.py \
    --pkl_dir=${PKL_DIR} \
    --out_path=${out_path} \
    --rescore_cfg=${rescore_config} \
    --K=${K} \
    --alpha=${alpha} \
    --beta=${beta} \
    --rescore=${rescore} \
    --test=${test}

