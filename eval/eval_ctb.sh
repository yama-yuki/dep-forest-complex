#!/bin/bash

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex

model_type=All
data_type=ctb5.1
data=ctb5.1

echo ${model_type}
echo ${data_type}
echo ${data}

BIAF_DIR=${PJ_DIR}/biaffine_forest/saves/ctb5.1
biaf_path=${BIAF_DIR}/test.conllx_1best.txt

GOLD_DIR=${PJ_DIR}/biaffine_forest/data/ctb5.1
gold_path=${GOLD_DIR}/test.conllx

EisnerK=16
K=3
A=01
B=01
PRED_DIR=${PJ_DIR}/outputs/pruned/unlabel2
pred_path=${PRED_DIR}/${model_type}/${data_type}/k${EisnerK}/rescore_${K}-${A}-${B}.conllu

vanilla_path=${PJ_DIR}/outputs/vanilla/${data_type}/k${EisnerK}/vanilla_${K}.conllu

cd ${PJ_DIR}/lib
python eval_unlabel.py \
    --pred_path=${pred_path} \
    --biaf_path=${biaf_path} \
    --gold_path=${gold_path} \
    --vanilla_path=${vanilla_path} \
    --eisner_k=${EisnerK} \
    --K=${K} 

