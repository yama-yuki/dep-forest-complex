#!/bin/bash

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex

model_type=V-V
data_type=wsj_sd_cophead
data=wsj_sd_cophead

echo ${model_type}
echo ${data_type}
echo ${data}

BIAF_DIR=${PJ_DIR}/biaffine_forest/saves/ptb_cophead
pred_path=${BIAF_DIR}/mytree_upos.conllu_16best.json

GOLD_DIR=${PJ_DIR}/biaffine_forest/data/wsj_sd_cophead
gold_path=${GOLD_DIR}/mytree_upos.conllu

cd ${PJ_DIR}/lib
python eval_unlabel.py \
    --pred_path=${pred_path} \
    --gold_path=${gold_path} \
    --nbest
