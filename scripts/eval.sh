#!/bin/bash

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
DATA_DIR=${PJ_DIR}/biaffine_forest/data/wsj_sd_cophead
PRED_DIR=${PJ_DIR}/outputs
pred_path=${PRED_DIR}/out.conllu
#data_path=${DATA_DIR}/test.conllu

data_path=${PJ_DIR}/test/test_data.conllu

python ../lib/eval.py \
    --pred_path=${pred_path} \
    --data_path=${data_path}

