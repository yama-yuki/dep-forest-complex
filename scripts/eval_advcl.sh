#!/bin/bash

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex

model_type=V-Any
data_type=mytree
if [ ${data_type} = mytree ] ; then
data=mytree_upos;
else
data=test
fi

echo ${model_type}
echo ${data_type}
echo ${data}

BIAF_DIR=${PJ_DIR}/biaffine_forest/saves/ptb_cophead
biaf_path=${BIAF_DIR}/${data}.conllu_1best.txt

GOLD_DIR=${PJ_DIR}/biaffine_forest/data/wsj_sd_cophead
gold_path=${GOLD_DIR}/${data}.conllu

EisnerK=4
K=4
A=03
B=01
PRED_DIR=${PJ_DIR}/outputs
pred_path=${PRED_DIR}/${model_type}/${data_type}/advcl/k${EisnerK}/rescore_${K}-${A}-${B}.conllu

cd ${PJ_DIR}/lib
python eval.py \
    --pred_path=${pred_path} \
    --biaf_path=${biaf_path} \
    --gold_path=${gold_path} \
    --eisner_k=${EisnerK} \

