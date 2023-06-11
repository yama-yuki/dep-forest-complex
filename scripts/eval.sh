#!/bin/bash

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
LIB_DIR=${PJ_DIR}/lib
#DATA_DIR=${PJ_DIR}/biaffine_forest/data/wsj_sd_cophead
#test_path=${DATA_DIR}/dev.conllu
DATA_DIR=${PJ_DIR}/biaffine_forest/saves/ptb_cophead
test_path=${DATA_DIR}/dev.conllu_1best.txt
PRED_DIR=${PJ_DIR}/outputs
pred_path=${PRED_DIR}/pred_4_1.conllu
#pred_path=${PRED_DIR}/pred_10_norescore.conllu


cd ${LIB_DIR}
python eval.py \
    --pred_path=${pred_path} \
    --test_path=${test_path} \


