#!/bin/bash

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex

model_type=V-V
data_type=ctb5.1
data=ctb5.1

echo ${model_type}
echo ${data_type}
echo ${data}

BIAF_DIR=${PJ_DIR}/biaffine_forest/saves/ctb5.1
pred_path=${BIAF_DIR}/test.conllx_16best.json

GOLD_DIR=${PJ_DIR}/biaffine_forest/data/ctb5.1
gold_path=${GOLD_DIR}/test.conllx

cd ${PJ_DIR}/lib
python eval_unlabel.py \
    --pred_path=${pred_path} \
    --gold_path=${gold_path} \
    --nbest
