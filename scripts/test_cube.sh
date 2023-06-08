#!/bin/bash

## paths
PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
PKL_DIR=${PJ_DIR}/biaffine_forest/pkl
RESCORE_DIR=${PJ_DIR}/rescore_module
rescore_config=${RESCORE_DIR}/rescore.cfg

## parameters
K=10
alpha=2
beta=1

## mode
rescore=False
test=True

cd ${PJ_DIR}
python forest_decoder.py \
    --pkl_dir=${PKL_DIR} \
    --rescore_cfg=${rescore_config} \
    --K=${K} \
    --alpha=${alpha} \
    --beta=${beta} \
    --rescore=${rescore} \
    --test=${test}

