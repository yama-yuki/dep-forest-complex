#!/bin/bash
PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
PKL_DIR=${PJ_DIR}/biaffine_forest/pkl
MODEL_DIR=${PJ_DIR}/rescore_module/models/

## finetuned model
model_name=bert-base-uncased_1_2_3e-5_32
test=True

cd ${PJ_DIR}

python forest_decoder.py \
    --pkl_dir=${PKL_DIR} \
    --model=${MODEL_DIR}/${model_name} \
    --test=${test}

